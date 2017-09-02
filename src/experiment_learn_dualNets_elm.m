function [] = experiment_learn_dualNets_elm()

close all;

category_lists = [];
%category_lists(1).name = 'ivy_56';
%category_lists(1).name = 'ivy_112';
category_lists(1).name = 'ivy_224';
%category_lists(1).name = 'ivy_448';
exp_type = 'texture';

for i = 1:length(category_lists)
    learn_dualNets(category_lists(i).name, exp_type);
end