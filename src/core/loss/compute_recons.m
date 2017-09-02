function recon_imgs = compute_recons(opts, config, net_cpu, imdb, getBatch, subset)
net = vl_simplenn_move(net_cpu, 'gpu');
recon_imgs = zeros([size(net.normalization.averageImage), numel(subset)], 'single');
numGpus = numel(opts.gpus) ;
for t=1:opts.batchSize:numel(subset)
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : batchEnd) ;
    [im, labels] = getBatch(imdb, batch);
    
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
        case {'semi'}
            dydz = gpuArray(single(1));
            net.layers{end}.class = labels ;
            res = vl_hae(net, gpuArray(im), dydz, [], [], 'conserveMemory', 1, 'cudnn', 1);
    end
    
    
    recon_imgs(:,:,:,batchStart:batchEnd) = gather(res(1).dzdx) * config.refsig^2;
end
clear net;
end