function generate_images(category, iteration, train_num)
% load('../../z_mats.mat');
% load(['./working/', category, '_',num2str(train_num) ,'_dense_em/layer_01_iter_', num2str(iteration), '_model.mat'])
load(['./working/', 'school_bus_1000_D23e-06_S20.0001_G20.0003/layer_01_iter_440_model.mat'])
for i = 1:100
    z_mats{i} = rand([1,1,100,train_num], 'single');
end

num_syn = 100;
net1 = vl_simplenn_move(net1, 'gpu') ;
net2 = vl_simplenn_move(net2, 'gpu') ;
numBatch = train_num/num_syn;

% config.T = 10;
syn_mats = [];

for i = 1:numBatch
    z = gpuArray(z_mats{i});

    %% without noise        
    syn_mat = vl_gan(net2, z, [], [], ...
        'accumulate', false, ...
        'disableDropout', false, ...
        'conserveMemory', true, ...
        'backPropDepth', +inf, ...
        'sync', false, ...
        'cudnn', true) ;
    syn_mat = syn_mat(end).x;
    syn_mats{i} = gather(syn_mat);
end

for TT = 10
    fprintf('Langevin steps: %d T=%d\n', TT, config.T);
    syn_mats1 = [];
    for i = 1:length(syn_mats)
        syn_mat = gpuArray(syn_mats{i});
        syn_mat = langevin_dynamics_fast(config, net1, syn_mat);
        syn_mat = gather(syn_mat);
        syn_mats{i} = syn_mat;

        syn_mat1 = [];
        for iImg = 1:100
            temp = syn_mat(:,:,:,iImg);
            temp = max(-1, min(1,temp)); % [-1,1]
            temp = single( uint8((temp+1)/2 * 255));
            gLow = min(temp(:));
            gHigh = max(temp(:));
            temp = (temp-gLow)/(gHigh-gLow);
           % syn_mat1 = cat(4, syn_mat1, temp);
            imwrite(temp, 'temp.png');
            syn_mat1 = cat(4, syn_mat1, imread('temp.png'));
        end
        syn_mats1{i} = syn_mat1;                     
    end
    save('syn_mats1','syn_mats1');

    
    for i = 1:numBatch
        syn_mat1 = syn_mats1{i};
%     for num = 1:100
%         imwrite(syn_mat1(:,:,:,num), ['./results/', category, '/',...
%         category, '_syn_', num2str((i-1)*100+num),'.JPEG']);
%         disp((i-1)*100+num);
%     end
    end
%     I_img = uint8(zeros([640, 640, 3]));
%     count = 0;
%     for x = 1:10
%         for y = 1:10
%             count = count + 1;
%             I_img((x-1)*64+1:x*64, (y-1)*64+1:y*64, :) = syn_mat1(:,:,:,count);
%         end
%     end
%     imwrite(I_img, ['./results/coop_', category,'/sample_coop_',...
%         category, '_', num2str(train_num), 'imgs_T', num2str(TT),'.png']);
            
end
net1 = vl_simplenn_move(net1, 'cpu') ;
net2 = vl_simplenn_move(net2, 'cpu') ;


end

