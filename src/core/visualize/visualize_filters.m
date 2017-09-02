function [] = visualize_filters(config, im, net, save_dir, start_layer)

if nargin < 2
    im = single(imread('/home/yanglu/Projects/autoEncoder/Image/codebook/ostrich/n01518878_94.JPEG'));
    im = imresize(im, net.normalization.imageSize(1:2));
    im = im - net.normalization.averageImage;
end

if nargin < 3
    model_file = '/home/yanglu/Projects/autoEncoder/release_1.3/working/codebook/ostrich_dense_net_frame_3_cd_e2e/layer_01_iter_2000_model.mat';
    load(model_file);
end

if nargin < 4
    save_dir = '/home/yanglu/Projects/autoEncoder/release_1.3/figure/filter/';
end

if nargin < 5
    start_layer = 1;
end

mean_img = net.normalization.averageImage;
net = vl_simplenn_move(net, 'gpu') ;


for l = 1:numel(net.layers)
    if strcmp(net.layers{l}.type, 'conv') == 1
        net.layers{l}.pad(:) = 0;
    end
end

brush = gpuArray(ones(size(im(:,:,:,1)), 'single'));

switch config.learn_scheme
    case {'hae_e2e', 'hae_l2l'}
        res = vl_hae(net, gpuArray(im), [],[],[],'conserveMemory', true);
    otherwise
        res = vl_simplenn(net, gpuArray(im), [], [], 'conserveMemory', true);
end


dydz_sz = size(res(end).x);
dydz = ones(dydz_sz, 'single');
[res, indicators] = vl_hae(net, gpuArray(im), gpuArray(dydz), [], [], ...
                    'conserveMemory', true, ...
                    'cudnn', true) ;

stride = 1;
fsz = 1;
space = 3;
for l = 1:numel(net.layers)
    net_l = net;
    net_l.layers = net_l.layers(1:l);
    if strcmp(net.layers{l}.type, 'conv') == 1
        if ~isempty(res(l+1).x);
            res_l = gather(res(l+1).x);
        else
            res_l = gather(res(l+2).x);
        end
        
        sz = size(net.layers{l}.weights{1});
        fsz = (sz(1)-1) * stride + fsz;
        num_filters = sz(4);
        
        
        stride = net.layers{l}.stride(1) * stride;
        
        if l >= start_layer
            dydz = zeros(size(res_l(:,:,:,1)), 'single');
            res_l = permute(res_l, [1, 2, 4, 3]);
            sz = size(res_l);
            res_l = reshape(res_l, [], num_filters);
            
            if size(res_l, 1) == 1
                idx = ones(size(res_l, 2), 1);
            else
                [~, idx] = max(res_l);
            end
            
            res_mean = mean(res_l, 1);
            
            if strcmp(config.model_type, 'sparse') == 1 && false
                num_filters = ceil(num_filters * config.sparse_level(l) / 100);
            end
            
            [~, filter_idx] = sort(res_mean, 'descend');
            filter_idx = filter_idx(1:num_filters);
            
            nr = ceil(sqrt(num_filters));
            canvas = zeros(nr * fsz + (nr-1)* space, ...
                nr * fsz + (nr-1)* space, 3, 'single');
            
            fr = 1;
            fc = 1;
            for i = filter_idx
                [idx_r, idx_c, idx_im] = ind2sub(sz, idx(i));%idx(i)
                dydz(:) = 0;
                dydz(idx_r, idx_c, i) = 1;
                indicators_l = indicators(1:numel(net_l.layers));
                for ll = 1:numel(net_l.layers)
                    if ~isempty(indicators_l{ll})
                        indicators_l{ll} = indicators_l{ll}(:,:,:,idx_im);
                    end
                end
                res_draw = vl_hae(net_l, brush, gpuArray(dydz), [], indicators_l, ...
                    'conserveMemory', true, ...
                    'cudnn', true) ;
                
                f_im = draw_filter(gather(res_draw(1).dzdx)*config.refsig^2, fsz, stride, idx_r, idx_c, mean_img);
                
                canvas(1+(fr-1)*(fsz+space):(fr-1)*(fsz+space) + fsz, ...
                    1+(fc-1)*(fsz+space):(fc-1)*(fsz+space) + fsz, :) = f_im;
                
                fc = fc + 1;
                if fc > nr
                    fc = 1;
                    fr = fr + 1;
                end
                
                clear res_draw;
            end
            
            % save filters
            imwrite(canvas, [save_dir, net.layers{l}.name, '.png']);
        end
    end
    
    if strcmp(net.layers{l}.type, 'pool') == 1
        if strcmp(net.layers{l}.method, 'inhibit') == 0
            stride = net.layers{l}.stride(1) * stride;
        end
    end
end

clear net;

function f_im = draw_filter(im, fsz, stride, idx_r, idx_c, mean_img)
start_r = stride * (idx_r - 1) + 1;
end_r = start_r + fsz - 1;
start_c = stride * (idx_c - 1) + 1;
end_c = start_c + fsz - 1;

f_im = im;
% mean_img = imresize(mean_img, [fsz, fsz]);
% im = abs(im);
% im = sum(im, 3);
% % determine starting column
% idx_vec = sum(im, 1);
% idx_c = find(idx_vec ~= 0);
% 
% % dertermine starting row
% idx_vec = sum(im, 2);
% idx_r = find(idx_vec ~= 0);

% f_im = uint8(f_im + mean_img);
f_im = f_im(start_r:end_r, start_c:end_c,:);

if nargin < 6
    gLow = min(reshape(f_im, [],1));
    gHigh = max(reshape(f_im, [],1));
    f_im = (f_im-gLow) / (gHigh - gLow);
else
%     f_im = uint8(f_im + mean_img);
%     f_im = single(f_im);
    gLow = min( reshape(f_im, [],1));
    gHigh = max(reshape(f_im, [],1));
    f_im = (f_im-gLow) / (gHigh - gLow);
%     f_im = f_im / 255;
end


