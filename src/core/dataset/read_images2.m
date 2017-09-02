function [img_mat, net] = read_images2(config, net)

img_file = [config.inPath, '/images.mat'];
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

files = files(1:100);
numImages = 0;

%% temp
%     img_file = '/media/vclagpu/Data1/yanglu/AutoEncoder/gan_1.3_0.5/working/codebook/celebA_dense_net_frame_gan_3_mask/images.mat';
%% end_temp

if exist(img_file, 'file') && ~config.force_learn
    load(img_file);
    numImages = size(img_mat, 4);
end

if numImages ~= length(files) || config.force_learn == true;
    img_mat = zeros([net.normalization.imageSize, length(files)], 'single');
    for iImg = 1:length(files)
        
        img = single(imread(fullfile(config.inPath, files(iImg).name)));
       
        if config.is_crop == true
            h = round((size(img, 1) - config.cropped_sz)/2);
         
            w = round((size(img, 2) - config.cropped_sz)/2);
        
            cropped_img = img(h:h+config.cropped_sz-1, w:w+config.cropped_sz-1, :);
          
            img = imresize(cropped_img, [config.sx,config.sy]);
        elseif config.is_preprocess == true
            % do rescaling 
            h = size(img, 1);
            w = size(img, 2);
            if (h < w)
                rescaled_img = imresize(img, [config.rescale_sz, config.rescale_sz * (w/h)]);
            else
                rescaled_img = imresize(img, [config.rescale_sz * (h/w), config.rescale_sz]);
            end
            
            % do random crop
            rescale_h = size(rescaled_img, 1);
            rescale_w = size(rescaled_img, 2);
            h1 = randi(rescale_h - config.sx, 1);
            w1 = randi(rescale_w - config.sy, 1);
            cropped_img = rescaled_img(h1:h1+config.sx-1, w1:w1+config.sy-1, :);
            img = cropped_img;
        else
            img = imresize(img, [config.sx,config.sy]);
        end
        
        img_mat(:,:,:,iImg) = 2*(img - min(img(:)))/(max(img(:))-min(img(:)))-1 ;
    end
    save(img_file, 'img_mat');
end

net.normalization.averageImage = ones(net.normalization.imageSize) ...
    * mean(img_mat(:));

for iImg = 1:length(files)
    img_mat(:,:,:,iImg) = img_mat(:,:,:,iImg) - net.normalization.averageImage;
end