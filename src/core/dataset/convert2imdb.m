function [imdb, fn] = convert2imdb(img_mat, masks)
% --------------------------------------------------------------------
if nargin < 2
   masks = []; 
end
numImages = size(img_mat, 4);
imdb.images.data = img_mat ;
imdb.images.set = ones(1, numImages);
imdb.images.set(10001:end) = 2;
imdb.meta.sets = {'train', 'val', 'test'} ;

if(~isempty(masks))
    imdb.images.masks = masks;
    fn = @(imdb,batch)getBatch_mask(imdb,batch);
else
    fn = @(imdb,batch)getBatch(imdb,batch);
end


end

function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = ones(1, numel(batch), 'single');
labels(1:6) = 1;
labels(7:end) = 2;
end

function [im, labels, mask] = getBatch_mask(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = ones(1, numel(batch), 'single');
labels(1:6) = 1;
labels(7:end) = 2;
mask = imdb.images.masks(:,:,:,batch);
end