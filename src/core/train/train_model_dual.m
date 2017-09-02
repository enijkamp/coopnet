function [net1, net2, config] = train_model_dual(config, net1, net2, imdb, getBatch, layer)

rate_list = logspace(-2, -4, 80)*100;
learningRate_array = repmat(rate_list , max(1,floor(config.nIteration / length(rate_list))),1); %logspace(-2, -4, 60) ones(1,60, 'single')
learningRate_array = reshape(learningRate_array, 1, []);


opts.batchSize = config.batch_size ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = config.gpus; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false;
opts.numFetchThreads = 8;
opts.cudnn = true ;
opts.weightDecay = 0.0001 ; %0.0001
opts.momentum = 0.5;
opts.memoryMapFile = fullfile(config.working_folder, 'matconvnet.bin') ;
opts.learningRate = reshape(learningRate_array, 1, []);

if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

opts.batchSize = min(opts.batchSize, numel(opts.train));
opts.numEpochs = config.nIteration;

net1 = initialize_momentum(net1);
net2 = initialize_momentum(net2);

interval = ceil(opts.numEpochs / 50);
SSD = zeros(opts.numEpochs, 1);

mean_img1 = net1.normalization.averageImage;
mean_img2 = net2.normalization.averageImage;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% -------------------------------------------------------------------------
%                           Train and validate
% -------------------------------------------------------------------------

model_file = [config.working_folder, num2str(layer, 'layer_%02d'), '_iter_',...
    num2str(opts.numEpochs) ,'_model.mat'];

if exist(model_file, 'file') && config.force_learn == false;
    load(model_file);
else
    h = figure;
    for epoch=1:opts.numEpochs
        fprintf('Iteration %d / %d\n', epoch, opts.numEpochs);
        
        learningRate = opts.learningRate(min(epoch, numel(opts.learningRate)));
        train = opts.train;
        
        [net1, net2, config, syn_mats, z_mats] = process_epoch_dual(opts, config, getBatch, epoch, train, learningRate, imdb, net1, net2);
        loss = compute_loss(opts, syn_mats, net2, z_mats);

        
        SSD(epoch) = loss;
        
        disp(['Loss: ', num2str(SSD(epoch))]);
        if mod(epoch - 1, interval) == 0 || epoch == opts.numEpochs
            
            syn_mat = syn_mats{1};
            draw_figures(config, syn_mat, epoch, mean_img1, SSD, layer, 'net1');
            
            z = z_mats{1};            
            generated_imgs = generate_imgs(opts, net2, z);
            draw_figures(config, generated_imgs, epoch, mean_img2, [], layer, 'net2');
            
            if mod(epoch, inf) == 0 || epoch == opts.numEpochs
                
                cell_idx = randperm(numel(z_mats), 1);
                z = z_mats{cell_idx};
                interpolator(config, net2, z, epoch);
                
                model_file = [config.working_folder, num2str(layer, 'layer_%02d'), '_iter_',...
                    num2str(epoch) ,'_model.mat'];
                save(model_file, 'net1', 'net2', 'z_mats', 'syn_mats', 'config');
                
                saveas(h, [config.working_folder, num2str(layer, 'layer_%02d_'), '_iter_',...
                    num2str(epoch) ,'_error.fig']);
                saveas(h, [config.working_folder, num2str(layer, 'layer_%02d_'), '_iter_',...
                    num2str(epoch) ,'_error.png'])
            end
        end
    end
end
end

function loss = compute_loss(opts, syn_mats, net_cpu, z_mats)
net = vl_simplenn_move(net_cpu, 'gpu') ;
loss = 0;
res = [];
for i=1:numel(syn_mats)
    syn_mat = syn_mats{i};
    z = z_mats{i};

    res = vl_gan(net, gpuArray(z), gpuArray(syn_mat), res, ...
        'accumulate', false, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'sync', opts.sync, ...
        'cudnn', opts.cudnn) ;
    loss = loss + gather( mean(reshape(sqrt((res(end).x - syn_mat).^2), [], 1)));
end
end

function imgs = generate_imgs(opts, net_cpu, z)
net = vl_simplenn_move(net_cpu, 'gpu') ;
res = vl_gan(net, gpuArray(z), [], [], ...
    'accumulate', false, ...
    'disableDropout', true, ...
    'conserveMemory', opts.conserveMemory, ...
    'backPropDepth', opts.backPropDepth, ...
    'sync', opts.sync, ...
    'cudnn', opts.cudnn) ;
imgs = gather(res(end).x);
end

function net = initialize_momentum(net)
for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        J = numel(net.layers{i}.weights) ;
        for j=1:J
            net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
        end
        if ~isfield(net.layers{i}, 'learningRate')
            net.layers{i}.learningRate = ones(1, J, 'single') ;
        end
        if ~isfield(net.layers{i}, 'weightDecay')
            net.layers{i}.weightDecay = ones(1, J, 'single') ;
        end
    end
end
end