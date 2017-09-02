function mask = generate_salt_pepper(sz, noise_level, mask_sz)


if numel(sz) == 4
    num_imgs = sz(4);
else
    num_imgs = 1;
end

% half_sz = floor(mask_sz / 2);
max_stride = mask_sz;

% faded layer
weights = gpuArray(ones([mask_sz, mask_sz, 3], 'single'));

mask = gpuArray(zeros(sz, 'single'));
res = gpuArray(randn(sz(1)-mask_sz+1, sz(2)-mask_sz+1, 1, num_imgs, 'single'));
sz_res = size(res);

dydz = gpuArray(zeros([ floor(sz_res(1) / max_stride), floor(sz_res(2) / max_stride), 1, num_imgs], 'single'));
dydz_sz = size(dydz);
idx = randperm(dydz_sz(1) * dydz_sz(2) * num_imgs);
idx = idx(1: max(1, floor(prod(dydz_sz) * noise_level)));
dydz(idx) = 1;

res_dydz = vl_nnpool(res, [mask_sz, mask_sz], dydz, ...
    'pad', [0,0,0,0], 'stride', max_stride, ...
    'method', 'max') ;

[mask, ~, ~] = vl_nnconv(mask, weights, gpuArray(zeros(1,1,'single')), ...
    res_dydz, 'pad', 0, 'stride', 1) ;
mask = gather(mask);
mask = mask ~= 0;