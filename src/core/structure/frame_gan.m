function net = frame_gan()
net.layers = [];
opts.scale = 2 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = true;
opts.addrelu = false;
opts.type = 'convt';
opts.leak = 0.2;

%% layer 1
layer_name = '1';
num_in = 1;
num_out = 512;
filter_sz = 5; 
upsample = 2; 
crop= [1,2,1,2];
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, upsample, crop);
net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s',layer_name));


%% layer 2
layer_name = '2';
num_in = 512;
num_out = 256;
filter_sz = 5;
upsample = 2;
crop = [1,2,1,2];
net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, upsample, crop);
net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s',layer_name));


%% layer 3
layer_name = '3';
num_in = 256;
num_out = 128; % 64 40
filter_sz = 5; %7 size: 34, 5
upsample = 2;%2 , 1
crop = [1,2,1,2];

net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, upsample, crop) ;
net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s',layer_name));

%% layer 4
layer_name = '4';
num_in = 128;
num_out = 64;
filter_sz = 5; %11
upsample = 2; %2, 8, 3
crop = [1,2,1,2];

net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, upsample, crop);
net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s',layer_name));

%% layer 5
opts.batchNormalization = false;
layer_name = '5';
num_in = 64;
num_out = 3;
filter_sz = 5; %fully
upsample = 2; %fully
crop = [1,2,1,2];%fully

net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, upsample, crop);
net.layers{end+1} = struct('type', 'tanh', 'name', sprintf('tanh%s',layer_name));

