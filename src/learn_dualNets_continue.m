function [net1, net2, config] = learn_dualNets_continue(category, siter, eiter)

rng(1);
num_imgs=1000;
learningTime = tic;
exp_type = 'object';
% fprintf('Learning category: %s\n', category);
Delta2 = 0.000003;
refsig2 = 0.0001;
gamma2 =  0.0003;
% for i=1:2
%     refsig2 = refsig2 / 10;
%     for j = 1:4
%         Delta2 = Delta2 / 10;
config = frame_config(category, 'em', exp_type, num_imgs, Delta2, refsig2, gamma2);

model_file = [config.working_folder, num2str(Delta2), '_', num2str(refsig2), '_iter_',...
    num2str(siter) ,'_model.mat'];
load(model_file);
config = frame_config(category, 'em', exp_type, num_imgs, Delta2, refsig2, gamma2);
config.nIteration = eiter;
%% Setup Network 2
config.z_sz = [1, 1, size(net2.layers{1}.weights{1}, 4)];

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

config.layer_sets2 = numel(net2.layers):-1:1;

config.sx = config.dydz_sz2(1);
config.sy = config.dydz_sz2(2);


%% Setup Network 1

img = randn(net1.normalization.imageSize, 'single');
net1 = vl_simplenn_move(net1, 'gpu') ;
res = vl_simplenn(net1, gpuArray(img));
net1 = vl_simplenn_move(net1, 'cpu');
config.dydz_sz1 = size(res(end).x);

config.layer_sets1 = numel(net1.layers):-1:1;

clear res;
clear img;

%% Step 2 create imdb
[imdb, getBatch, net1] = create_imdb(config, net1, 1000);

%% Step 4: training
[net1, net2, config] = train_model_dual_continue(config, net1, net2, imdb, getBatch, 1, siter);


learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);