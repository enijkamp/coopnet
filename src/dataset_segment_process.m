function dataset_segment_process(dataset_path, train_num, test_num)
category_lists = dir([dataset_path, '/', '*']);
valid_category = true(1, length(category_lists));
for i = 1:length(category_lists)
    if category_lists(i).name(1) == '.' || ~category_lists(i).isdir
       valid_category(i) = false; 
    end
end
category_lists = category_lists(valid_category);
trainset = [dataset_path, '_train_', num2str(train_num)];
testset = [dataset_path, '_test_', num2str(train_num)];

for ii = 1:length(category_lists)
files = dir([dataset_path, '/', category_lists(ii).name ,'/*.JPEG']);
train_cat = [trainset, '/',  category_lists(ii).name];
test_cat = [testset, '/',  category_lists(ii).name];
mkdir (train_cat)
mkdir (test_cat)
for j = 1:train_num + test_num
    if j <= train_num
        copyfile([dataset_path,'/', category_lists(ii).name, '/',files(j).name], ...
            [trainset, '/', category_lists(ii).name]);
    else
        copyfile([dataset_path, '/',category_lists(ii).name,'/', files(j).name], ...
            [testset, '/', category_lists(ii).name]);
    end   
end
end
end