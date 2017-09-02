function mask = generate_masks(sz, mask_sz)


if numel(sz) == 4
    num_imgs = sz(4);
else
    num_imgs = 1;
end

% half_sz = floor(mask_sz / 2);
max_stride = 1;

% faded layer
weights = gpuArray(ones([mask_sz, mask_sz, 3], 'single'));

mask = gpuArray(zeros(sz, 'single'));
res = gpuArray(randn(sz(1)-mask_sz+1, sz(2)-mask_sz+1, 1, num_imgs, 'single'));
sz_res = size(res);

dydz = gpuArray(zeros([ floor(sz_res(1) / max_stride), floor(sz_res(2) / max_stride), 1, num_imgs], 'single'));
dydz_sz = size(dydz);
while numel(dydz_sz) < 4
    dydz_sz = [dydz_sz, 1];
end
rand_y = randi([1, dydz_sz(1)], 1, dydz_sz(end));
rand_x = randi([1, dydz_sz(2)], 1, dydz_sz(end));
idx = sub2ind(dydz_sz, rand_y, rand_x, ones(1, dydz_sz(end)), 1:dydz_sz(end));
dydz(idx) = 1;

[mask, ~, ~] = vl_nnconv(mask, weights, gpuArray(zeros(1,1,'single')), ...
    dydz, 'pad', 0, 'stride', 1) ;

% res_dydz = vl_nnpool(res, [mask_sz, mask_sz], dydz, ...
%     'pad', [0,0,0,0], 'stride', max_stride, ...
%     'method', 'max') ;
% 
% [mask, ~, ~] = vl_nnconv(mask, weights, gpuArray(zeros(1,1,'single')), ...
%     res_dydz, 'pad', 0, 'stride', 1) ;
mask = gather(mask);
mask = mask ~= 0;