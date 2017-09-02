% function net = convNet_texture()
% net.layers = [];
% opts.scale = 1 ;
% opts.initBias = 0.1 ;
% opts.weightDecay = 1 ;
% opts.weightInitMethod = 'gaussian' ;
% opts.model = 'alexnet' ;
% opts.batchNormalization = false;
% opts.addrelu = true;
% 
% 
% %% layer 1
% layer_name = '1';
% num_in = 3;
% num_out = 100;
% filter_sz = 15;
% stride = 3;
% pad = 7;
% net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
% 
% 
% %% layer2
% layer_name = '2';
% num_in = num_out;
% num_out = 70;
% filter_sz = 9; 
% stride = 1;
% pad = 4;
% net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad) ;
% 
% 
% %% layer3
% layer_name = '3';
% num_in = num_out;
% num_out = 30;
% filter_sz = 7;
% stride = 1;
% pad = 3;
% net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);

function net = convNet_texture()
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
num_out = 200;
filter_sz = 11;
stride = 2;
pad_sz = floor(filter_sz/2);
pad = ones(1,4)*pad_sz;
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);


%% layer2
layer_name = '2';
num_in = num_out;
num_out = 100;
filter_sz = 7;
stride = 2;
pad_sz = floor(filter_sz/2);
pad = ones(1,4)*pad_sz;
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad) ;


%% layer3
layer_name = '3';
num_in = num_out;
num_out = 1;
filter_sz = 8;
stride = 1; 
pad_sz = 0;
pad = ones(1,4)*pad_sz;
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
