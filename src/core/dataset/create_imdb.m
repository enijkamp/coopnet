function [imdb, getBatch, net] = create_imdb(config, net)

switch config.datatype
    case 'small'
        [img_mat, net] = read_images(config, net);
        [imdb, getBatch] = convert2imdb(img_mat);
    case 'celebA'
        [img_mat, net] = read_images2(config, net);
        [imdb, getBatch] = convert2imdb(img_mat);
    case 'celebB'
        [img_mat, masks] = read_images3(config, net);
        [imdb, getBatch] = convert2imdb(img_mat, masks);
    case 'large'
        opts.dataDir = config.inPath;
        opts.lite = false ;
        opts.imdbPath = [config.working_folder, 'imdb.mat'];
        opts.subCat = [];
        opts.subCat = {'CatHead'}; %{'CatHead'};
        bopts = net.normalization ;
        bopts.numThreads = 12;
        bopts.transformation = 'stretch';
        bopts.averageImage = [];
        
        if exist(opts.imdbPath, 'file') && ~config.force_learn
            imdb = load(opts.imdbPath);
        else
            imdb = large_data_setup('dataDir', opts.dataDir,...
                'lite', opts.lite, 'subCat', opts.subCat);
            save(opts.imdbPath, '-struct', 'imdb');
        end
        
        imageStatsPath = fullfile(config.working_folder, 'imageStats.mat') ;
        if exist(imageStatsPath, 'file')
            load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
        else
            [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
            save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
        end
        
        net.normalization.averageImage = ones(size(averageImage), 'single') * mean(rgbMean);
        bopts.averageImage = net.normalization.averageImage;
        getBatch = getBatchSimpleNNWrapper(bopts);
end


function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 1: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
    temp = fn(imdb, batch) ;
    z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
    n = size(z,2) ;
    avg{t} = mean(temp, 4) ;
    rgbm1{t} = sum(z,2)/n ;
    rgbm2{t} = z*z'/n ;
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
