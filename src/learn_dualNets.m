function [net1, net2, config] = learn_dualNets(category, exp_type)

rng(1);

learningTime = tic;

fprintf('Learning category: %s\n', category);

config = frame_config(category, 'em', exp_type);


%% Setup Network 2
switch exp_type
    case 'texture'
        net2 = frame_gan();
        config.z_sz = [7, 7, size(net2.layers{1}.weights{1}, 4)];
    case 'object'
        net2 = frame_gan_object();
        config.z_sz = [1, 1, size(net2.layers{1}.weights{1}, 4)];
end

net2.z_sz = config.z_sz;

config.dydz_sz2 = [config.z_sz(1:2), 3];
for l = 1:numel(net2.layers)
    if strcmp(net2.layers{l}.type, 'convt')
        f_sz = size(net2.layers{l}.weights{1});
        crops = [net2.layers{l}.crop(1)+net2.layers{l}.crop(2), ...
            net2.layers{l}.crop(3)+net2.layers{l}.crop(4)];
        config.dydz_sz2(1:2) = net2.layers{l}.upsample.*(config.dydz_sz2(1:2) - 1) ...
            + f_sz(1:2) - crops;
    end
end
net2.dydz_sz = config.dydz_sz2;

z = randn(config.z_sz, 'single');
net2 = vl_simplenn_move(net2, 'gpu');
res = vl_gan(net2, gpuArray(z));
net2 = vl_simplenn_move(net2, 'cpu');

net2.numFilters = zeros(1, length(net2.layers));
for l = 1:length(net2.layers)
    if isfield(net2.layers{l}, 'weights')
        sz = size(res(l+1).x);
        net2.numFilters(l) = sz(1) * sz(2);
    end
end

config.layer_sets2 = numel(net2.layers):-1:1;

net2.normalization.imageSize = config.dydz_sz2;
net2.normalization.averageImage = zeros(config.dydz_sz2, 'single');
config.sx = config.dydz_sz2(1);
config.sy = config.dydz_sz2(2);
clear z;


%% Setup Network 1
switch exp_type
    case 'texture'
        net1 = convNet_texture();
    case 'object'
        net1 = convNet_object(config);
end

net1.normalization.imageSize = [config.sx, config.sy, 3];
net1.normalization.averageImage = net2.normalization.averageImage;

img = randn(net1.normalization.imageSize, 'single');
net1 = vl_simplenn_move(net1, 'gpu') ;
res = vl_simplenn(net1, gpuArray(img));
net1 = vl_simplenn_move(net1, 'cpu');
config.dydz_sz1 = size(res(end).x);

net1.numFilters = zeros(1, length(net1.layers));
for l = 1:length(net1.layers)
    if isfield(net1.layers{l}, 'weights')
        sz = size(res(l+1).x);
        net1.numFilters(l) = sz(1) * sz(2);
    end
end

config.layer_sets1 = numel(net1.layers):-1:1;

clear res;
clear img;

%% Step 2 create imdb
[imdb, getBatch, net1] = create_imdb(config, net1);

%% Step 4: training
[net1, net2, config] = train_model_dual(config, net1, net2, imdb, getBatch, 1);


learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);