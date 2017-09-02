function net = convNet_object(config)
net.layers = [];
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = true;


%% layer 1
layer_name = '1';
num_in = 3;
num_out = 64;
filter_sz = 4;
stride = 2;
pad = 2;
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);


%% layer2
layer_name = '2';
num_in = num_out;
num_out = 128;
filter_sz = 2; 
stride = 1;
pad = 2;
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad, 0.5) ;

img = randn([config.sx, config.sy, 3], 'single');
net = vl_simplenn_move(net, 'gpu') ;
res = vl_simplenn(net, gpuArray(img));
net = vl_simplenn_move(net, 'cpu');
dydz_sz = size(res(end).x);


%% layer top
numFilters = 2; %% 
stride = 1;
pad_sz = 0;
pad = ones(1,4)*pad_sz;

opts.addrelu = false;

layer_name = '3_1';
net = add_cnn_block(net, opts, layer_name, dydz_sz(1), dydz_sz(1), num_out, numFilters, stride, pad, 0.2857);


