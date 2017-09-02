function [img_mat, masks] = read_images3(config, net)

img_file = [config.working_folder, 'images.mat'];
files = dir([config.inPath '/*.jpg']);


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
    load([config.inPath, '/masks.mat'])
    numImages = size(img_mat, 4);
end

if numImages ~= length(files) || config.force_learn == true;
    load([config.inPath, '/masks.mat'])
    img_mat = zeros([net.normalization.imageSize, length(files)], 'single');
    for iImg = 1:length(files)
        
        img = single(imread(fullfile(config.inPath, files(iImg).name)));
        img_mat(:,:,:,iImg) = 2*(img - min(img(:)))/(max(img(:))-min(img(:)))-1 ;
    end
    save(img_file, 'img_mat');
end