function net = add_cnn_block(net, opts, id, h, w, in, out, stride, pad, learning_rate)
% --------------------------------------------------------------------
if nargin < 10
    learning_rate = 1;
end

if ~isfield(opts, 'type')
    opts.type = 'conv';
end

info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
    name = 'fc' ;
elseif strcmp(opts.type, 'conv') == true
    name = 'conv' ;
else
    name = 'convt';
end
switch opts.type
    case 'conv'
        net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
            'weights', {{gpuArray(init_weight(opts, h, w, in, out, 'single')), gpuArray(zeros(1, out, 'single'))}}, ...
            'stride', [stride, stride], ...
            'pad', pad, ...
            'learningRate', learning_rate*[1, 2], ...
            'weightDecay', [opts.weightDecay 0]) ;
    case 'convt'
        net.layers{end+1} = struct('type', 'convt', 'name', sprintf('%s%s', name, id), ...
            'weights', {{gpuArray(init_weight(opts, h, w, out, in, 'single')), gpuArray(zeros(1, 1, out, 'single'))}}, ...
            'upsample', [stride, stride], ...
            'crop', pad, ...
            'numGroups', 1, ...
            'learningRate', learning_rate*[1, 2], ...
            'weightDecay', [opts.weightDecay 0]) ;
        
end
if opts.batchNormalization
    switch opts.type
        case 'conv'
            net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d',id), ...
                'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single')}}, ...
                'learningRate', learning_rate*[2 1], ...
                'weightDecay', [0 0]) ;
        case 'convt'
            net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d',id), ...
                'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single')}}, ...
                'learningRate', learning_rate*[2 1], ...
                'weightDecay', [0 0]) ;
    end
end
if opts.addrelu
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id));
end
end


function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    case 'zero'
        weights = zeros(h, w, in, out, type);
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
end