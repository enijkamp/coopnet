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