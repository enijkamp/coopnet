function loss = computerLoss(opts, config, imdb, getBatch, subset, net_cpu)

numGpus = numel(opts.gpus) ;
net = vl_simplenn_move(net_cpu, 'gpu') ;

loss = 0;

for t=1:opts.batchSize:numel(subset)
    for s=1:opts.numSubBatches
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        [im, labels] = getBatch(imdb, batch);
        
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
        
        if numGpus >= 1
            im = gpuArray(im) ;
        end
               
        switch config.learn_scheme
            case {'hae_l2l', 'hae_e2e'}
                % training images
                numImages = size(im, 4);
                net.layers{end}.class = labels ;
                
                dydz = gpuArray(zeros(config.dydz_sz, 'single'));
                dydz(net.filterSelected) = net.selectedLambdas;
                dydz = repmat(dydz, 1, 1, 1, numImages);
                res = vl_hae(net, gpuArray(im), dydz, [], [], 'conserveMemory', 1, 'cudnn', 1);
                recon_imgs = res(1).dzdx * config.refsig^2;
            case {'semi'}
                dydz = gpuArray(single(1));
                net.layers{end}.class = labels ;
                res = vl_hae(net, gpuArray(im), dydz, [], [], 'conserveMemory', 1, 'cudnn', 1);
                recon_imgs = res(1).dzdx * config.refsig^2;
            case {'cd_mask_l2l', 'cd_mask_e2e'}
                numImages = size(im, 4);
                dydz = gpuArray(zeros(config.dydz_sz, 'single'));
                dydz(net.filterSelected) = net.selectedLambdas;
                dydz = repmat(dydz, 1, 1, 1, numImages);
                
                mask = generate_salt_pepper(config, size(im));
%                 mask(:) = true;
                syn_mat = im;
                syn_mat(mask) = 0;
                
                for tt = 1:config.T
                    res = vl_simplenn(net, gpuArray(im), dydz, [], 'conserveMemory', 1, 'cudnn', 1);
                
                    temp_mat = config.Delta^2/2 * (res(1).dzdx - syn_mat / config.refsig /config.refsig) + ...
                        config.Delta * gpuArray(randn(size(syn_mat), 'single'));
                    
                    syn_mat(mask) = syn_mat(mask) + temp_mat(mask);
                end
                recon_imgs = gather(syn_mat);
            otherwise
                numImages = size(im, 4);
                dydz = gpuArray(zeros(config.dydz_sz, 'single'));
                dydz(net.filterSelected) = net.selectedLambdas;
                dydz = repmat(dydz, 1, 1, 1, numImages);
                res = vl_simplenn(net, gpuArray(im), dydz, [], 'conserveMemory', 1, 'cudnn', 1);
                recon_imgs = res(1).dzdx * config.refsig^2;
        end

        loss = loss + gather(mean(reshape(sqrt((recon_imgs - im).^2), [], 1)));
    end
end
loss = loss / ceil(numel(subset)/ opts.batchSize);
clear net;
end