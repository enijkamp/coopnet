function loss = test_model_sampling(net, config, getBatch, imdb)
mean_img = net.normalization.averageImage;
net = vl_simplenn_move(net, 'gpu') ;

subset = config.test;

num_batch = ceil(numel(subset)/config.BatchSize);

loss.noise = zeros(1, num_batch);
loss.denoise = loss.noise;

interval = ceil(config.nIterSampling / 70);
idx = randi([1, num_batch], 1, config.nIterSampling);
idx(1) = 1;

for t=1:config.BatchSize:numel(subset)
    cur_batch = floor(t/config.BatchSize)+1;
    
    fprintf('sampling: batch %3d/%3d:\n', ...
        cur_batch, num_batch) ;
    
    batchStart = t + (labindex-1);
    batchEnd = min(t+config.BatchSize-1, numel(subset)) ;
    batch = subset(batchStart : batchEnd) ;
    
    config.nTileRow = ceil(sqrt(numel(batch)));
    config.nTileCol = config.nTileRow;
    
    im = getBatch(imdb, batch);
    syn_mat = im;
    
    switch config.sample_type 
        case 'denoise'
            config.cur_layer = 3;
            [mask, noise_level] = generate_salt_pepper(config, size(syn_mat), [config.noise_level, config.noise_level], [1,1,1]);
            syn_mat(mask) = 0;
        case 'inpainting'
%             mask = false(size(im));
%             mask(60:140,60:140,:,:) = true;
%             mask(10:50,10:50,:,:) = true;
%             mask(10:50,150:190,:,:) = true;
%             mask(150:190,10:50,:,:) = true;
%             mask(150:190,150:190,:,:) = true;
%             syn_mat(mask) = 0;
            config.cur_layer = 3;
            [mask, noise_level] = generate_salt_pepper(config, size(syn_mat), [config.noise_level, config.noise_level], [10,10,10]);
            syn_mat(mask) = 0;
    end
    
    numImages = size(im, 4);
    
    ori_img = uint8(im + repmat(mean_img, 1, 1, 1, numImages));
    noise_img = uint8(syn_mat + repmat(mean_img, 1, 1, 1, numImages));
    loss.noise(cur_batch) = psnr(noise_img, ori_img);
    fprintf('Before denoising/inpainting, the psnr is %.2f\n', loss.noise(cur_batch));
        
    dydz = gpuArray(zeros(config.dydz_sz, 'single'));
    dydz(net.filterSelected) = net.selectedLambdas;
    dydz = repmat(dydz, 1, 1, 1, numImages);
    res = [];
    
    for iter = 1:config.nIterSampling
        fprintf('Langevin dynamics sampling iteration %d: ', iter);
        syn_mat = gpuArray(syn_mat);
        res = vl_simplenn(net, syn_mat, gpuArray(dydz), res, ...
            'accumulate', false, ...
            'disableDropout', true, ...
            'conserveMemory', true, ...
            'backPropDepth', inf) ;
        
        temp_mat = config.Delta^2/2 * (res(1).dzdx - syn_mat / config.refsig /config.refsig) + ...
            config.Delta * gpuArray(randn(size(syn_mat), 'single'));
        syn_mat(mask) = syn_mat(mask) + temp_mat(mask);
        syn_mat = gather(syn_mat);
        
        if (mod(iter - 1, interval) == 0 || iter == config.nIterSampling || iter == 1) && idx(iter) == cur_batch
            % save samples
            draw_figures(config, syn_mat, iter, mean_img, [], [], 1, [], [], false);
        end
        
        denoise_img = uint8(syn_mat + repmat(mean_img, 1, 1, 1, numImages));
        fprintf('the psnr is %.2f\n', psnr(denoise_img, ori_img));
    end
    denoise_img = uint8(syn_mat + repmat(mean_img, 1, 1, 1, numImages));
    loss.denoise(cur_batch) = psnr(denoise_img, ori_img);
    
    [~, syn_mat_norm] = convert_syns_mat(config, mean_img, syn_mat);
    for i = 1:size(syn_mat_norm, 4)
        switch config.sample_type
            case 'denoise'
                imwrite(noise_img(:,:,:,i), [config.figure_folder, num2str((cur_batch-1)*config.BatchSize+i, '%04d'), num2str(noise_level, '_%.2f_noise.png')]);
                imwrite(syn_mat_norm(:,:,:,i), [config.figure_folder, num2str((cur_batch-1)*config.BatchSize+i, '%04d'), num2str(noise_level, '_%.2f_denoise.png')]);
                imwrite(ori_img(:,:,:,i), [config.figure_folder, num2str((cur_batch-1)*config.BatchSize+i, '%04d_ori.png')]);
            case 'inpainting'
                imwrite(noise_img(:,:,:,i), [config.figure_folder, num2str((cur_batch-1)*config.BatchSize+i, '%04d_blank.png')]);
                imwrite(syn_mat_norm(:,:,:,i), [config.figure_folder, num2str((cur_batch-1)*config.BatchSize+i, '%04d_inpainting.png')]);
                imwrite(ori_img(:,:,:,i), [config.figure_folder, num2str((cur_batch-1)*config.BatchSize+i, '%04d_ori.png')]);
        end
    end
end