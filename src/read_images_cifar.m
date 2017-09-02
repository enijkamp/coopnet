function [img_mat, net] = read_images_cifar(config, net, num_imgs)

% img_file = [config.inPath, 'images.mat'];
files = dir([config.inPath '*.mat']);

if isempty(files)
    fprintf('error: No training images are found\n');
    keyboard;
end

numImages = 1;
img_mat = zeros([ config.sx, config.sy, 3, 5000]);
for i = 1:length(files)
% if exist(files(i).name, 'file') 
    sets = load([config.inPath files(i).name]);
    imgs = sets.data;
    label = sets.labels;
    for j = 1: size(imgs, 1)
        if label(j) == 1
            img = zeros([32,32,3]);
            for channel = 1:3
            img(:,:,channel) = reshape(imgs(j,1024*(channel-1)+1:1024*channel),[32, 32])';
            end
            img = single(imresize(img,[config.sx, config.sy]));
            min_val = min(img(:));
            max_val = max(img(:));
            img_mat(:,:,:,numImages) = (img - min_val) / (max_val - min_val)*2 - 1;
            numImages = numImages + 1;
        end
    end
    fprintf('read and process batch %d / %d\n', i, length(files))
%     numImages = numImages + size(img_mat, 1);
% end
    img_mat = single(img_mat);
end
save('cat1', 'img_mat');

% files = files(1:num_imgs);
% if numImages ~= length(files) || config.force_learn == true;
%     img_mat = zeros([net.normalization.imageSize, length(files)], 'single');
%     for iImg = 1:length(files)
%         fprintf('read and process images %d / %d\n', iImg, length(files))
% %         img = single(imread(fullfile(config.inPath, files(iImg).name)));
% %         img = imresize(img, [config.sx,config.sy]);
% %         min_val = min(img(:));
% %         max_val = max(img(:));
%         if length(size(img)) == 2
%             img_mat(:,:,1,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
%             img_mat(:,:,2,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
%             img_mat(:,:,3,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
%             disp('!');
%         else
% %             img_mat(:,:,:,iImg) = (img - min_val) / (max_val - min_val)*2 - 1;
%         end
%     end
%     save(img_file, 'img_mat');
% end

% net.normalization.averageImage = ones(net.normalization.imageSize) ...
%     * mean(img_mat(:));
% 
% for iImg = 1:length(files)
%     img_mat(:,:,:,iImg) = img_mat(:,:,:,iImg) - net.normalization.averageImage;
% end
