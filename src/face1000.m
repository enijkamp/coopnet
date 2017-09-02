src_dir = '../../data/celebA/';
dst_dir = '../../data/face1000/';

img_list = dir([src_dir, '*.jpg']);

idx = numel(img_list);
idx = randperm(idx, 1000);

for i = idx
    cmd = sprintf('cp %s%s %s%s', src_dir, img_list(i).name, dst_dir, img_list(i).name);
    system(cmd);
end