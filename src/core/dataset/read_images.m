function [img_mat, net] = read_images(config, net, num_imgs)

img_file = [config.working_folder, 'images.mat'];
files = dir([config.inPath '*.jpg']);

if isempty(files)
   files = dir([config.inPath '*.JPEG']); 
end
if isempty(files)
   files = dir([config.inPath '*.png']); 
end

if isempty(files)
    fprintf('error: No training images are found\n');
    keyboard;
end

numImages = 0;

if exist(img_file, 'file') && ~config.force_learn
    load(img_file);
    numImages = size(img_mat, 4);
end

files = files(1:num_imgs);
if numImages ~= length(files) || config.force_learn == true;
    img_mat = zeros([net.normalization.imageSize, length(files)], 'single');
    for iImg = 1:length(files)
        fprintf('read and process images %d / %d\n', iImg, length(files))
        img = single(imread(fullfile(config.inPath, files(iImg).name)));
        img = imresize(img, [config.sx,config.sy]);
        min_val = min(img(:));
        max_val = max(img(:));
        if length(size(img)) == 2
            img_mat(:,:,1,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
            img_mat(:,:,2,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
            img_mat(:,:,3,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
            disp('!');
        else
            img_mat(:,:,:,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
        end
    end
    save(img_file, 'img_mat');
end

% net.normalization.averageImage = ones(net.normalization.imageSize) ...
%     * mean(img_mat(:));
% 
% for iImg = 1:length(files)
%     img_mat(:,:,:,iImg) = img_mat(:,:,:,iImg) - net.normalization.averageImage;
% end
