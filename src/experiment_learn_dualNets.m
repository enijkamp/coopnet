function [] = experiment_learn_dualNets()
close all;
category_path = '../Image/texture/';
category_lists = dir([category_path, '*']);
valid_category = true(1, length(category_lists));
for i = 1:length(category_lists)
    if category_lists(i).name(1) == '.' || ~category_lists(i).isdir
       valid_category(i) = false; 
    end
end

category_lists = category_lists(valid_category);
category_lists = [];
category_lists(1).name = 'noodle';
exp_type = 'texture';

for ii = 1:length(category_lists)
    learn_dualNets(category_lists(ii).name, exp_type);
end

% experiment with simple category
category_list = { 'lemon'};
for ii = 1:length(category_list)
    for num_imgs = [300, 100]
        learn_dualNets(category_list{ii}, 'object', num_imgs);
    end
end

% generate images for simple category
category = 'balloon';
mkdir(['./results/coop_', category]);
num_imgs = [1000, 700, 500, 300, 100];
iter = [200, 350, 350, 350, 1000];
for i = 1:length(iter)
    generate_images(category, iter(i), num_imgs(i));
end









