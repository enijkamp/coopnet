function  [net1, net2, config, syn_mats, z_mats] = process_epoch_dual(opts, config, getBatch, epoch, subset, learningRate, imdb, net1, net2)
% -------------------------------------------------------------------------

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
net1 = vl_simplenn_move(net1, 'gpu') ;
net2 = vl_simplenn_move(net2, 'gpu') ;

mmap = [] ;
num_syn = config.nTileRow * config.nTileCol;

dydz_syn = gpuArray(ones(config.dydz_sz1, 'single'));
dydz_syn = repmat(dydz_syn, 1, 1, 1, config.nTileRow*config.nTileCol);
res1 = [];
res_syn = [];
res2 = [];
training = true;

num_cell = ceil(numel(subset) / opts.batchSize) * numel(opts.gpus);
syn_mats = cell(1, num_cell);
z_mats = cell(1, num_cell);

for t=1:opts.batchSize:numel(subset)
    fprintf('training: epoch %02d: batch %3d/%3d: \n', epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    batchTime = tic ;
    numDone = 0 ;
    
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        im = getBatch(imdb, batch) ;
        
        if opts.prefetch
            if s==opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            getBatch(imdb, nextBatch) ;
        end
        
        im = gpuArray(im);
        
        cell_idx = (ceil(t / opts.batchSize) - 1) * numlabs + labindex;
        % Step 1: Inference Network 2 -- generate Z
        z_mats{cell_idx} = randn([config.z_sz, num_syn], 'single');
        z = gpuArray(z_mats{cell_idx});
        syn_mat = vl_gan(net2, z, [], [],...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
        syn_mat = syn_mat(end).x;        
        
        % Step 2: Inference Network 1 -- generate synthesized images           
        syn_mat = langevin_dynamics_fast(config, net1, syn_mat);
        syn_mats{cell_idx} = gather(syn_mat);
        
        % Step 3: Learning Net1
        numImages = size(im, 4);
        dydz1 = gpuArray(ones(config.dydz_sz1, 'single'));
        dydz1 = repmat(dydz1, 1, 1, 1, numImages);
        
        res1 = vl_simplenn(net1, im, dydz1, res1, ...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn);
        
        res_syn = vl_simplenn(net1, syn_mat, dydz_syn, res_syn, ...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn);
        
        % Step 4: Learning Net2
        res2 = vl_gan(net2, z, syn_mat, res2, ...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
        
        numDone = numDone + numel(batch) ;
    end
    
    net1 = accumulate_gradients1(opts, learningRate * config.Gamma1, batchSize, net1, res1, res_syn, config);
    net2 = accumulate_gradients2(opts, learningRate * config.Gamma2, batchSize, net2, res2, config);
   
    fprintf('max inferred z is %.2f, min inferred z is %.2f, and std is %.2f\n', max(z(:)), min(z(:)), config.real_ref)
    
    % print learning statistics
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' [%d/%d]', numDone, batchSize);
    fprintf('\n') ;
end

real_ref = std(z(:));
config.real_ref = real_ref;

net1 = vl_simplenn_move(net1, 'cpu') ;
net2 = vl_simplenn_move(net2, 'cpu') ;
end


% -------------------------------------------------------------------------
function [net, res] = accumulate_gradients2(opts, lr, batchSize, net, res, config, mmap)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets2;

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        % accumualte from multiple labs (GPUs) if needed
        if nargin >= 7
            tag = sprintf('l%d_%d',l,j) ;
            tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
            for g = setdiff(1:numel(mmap.Data), labindex)
                tmp = tmp + mmap.Data(g).(tag) ;
            end
            res(l).dzdw{j} = res(l).dzdw{j} + tmp;
        end
        
        if isfield(net.layers{l}, 'weights')
            % gradient descent
            gradient_dzdw = (1 / batchSize) * (1 / config.s / config.s)* res(l).dzdw{j};
            
            max_val = max(abs(gradient_dzdw(:)));
            
            if max_val > config.cap2;
                gradient_dzdw = gradient_dzdw / max_val * config.cap2;
            end
  
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            %             net.layers{l}.momentum{j} = gradient_dzdw;
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR *net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+2, length(res));
                fprintf('Net2: layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
end

function [net, loss] = accumulate_gradients1(opts, lr, batchSize, net, res, res_syn, config, mmap)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets1;
num_syn = config.nTileRow * config.nTileCol;
% opts.momentum = 0;
% opts.weightDecay = 0;

loss = [];
for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        % accumualte from multiple labs (GPUs) if needed
        if nargin >= 8
            tag = sprintf('l%d_%d',l,j) ;
            tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
            for g = setdiff(1:numel(mmap.Data), labindex)
                tmp = tmp + mmap.Data(g).(tag) ;
            end
            res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
            
            if ~isempty(res_syn)
                tag = sprintf('syn_l%d_%d',l,j) ;
                tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
                for g = setdiff(1:numel(mmap.Data), labindex)
                    tmp = tmp + mmap.Data(g).(tag) ;
                end
                res_syn(l).dzdw{j} = res_syn(l).dzdw{j} + tmp ;
            end
        end
        
        if isfield(net.layers{l}, 'weights')
            gradient_dzdw = ((1 / batchSize) * res(l).dzdw{j} -  ...
                (1 / num_syn) * res_syn(l).dzdw{j}) / net.numFilters(l);
            if max(abs(gradient_dzdw(:))) > config.cap1 %10
                gradient_dzdw = gradient_dzdw / max(abs(gradient_dzdw(:))) * config.cap1;
            end
            
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            loss = [loss, gather(mean(abs(gradient_dzdw(:))))];
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR *net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+1, length(res));
                fprintf('Net1: layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
loss = mean(loss);
end

function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
    for j=1:numel(res(i).dzdw)
        format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
    end
end

format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname, 'file') && (labindex == 1)
    f = fopen(fname,'wb') ;
    for g=1:numGpus
        for i=1:size(format,1)
            fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
        end
    end
    fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;
end

function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
    for j=1:numel(res(i).dzdw)
        mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
    end
end
end
 