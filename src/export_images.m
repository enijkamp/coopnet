load('./working/imagenet_10category_500_dense_em/layer_01_iter_1000_model.mat');
syn_mats1 = [];

for c = 1:length(syn_mats)
    syn_mat = syn_mats{c};
    syn_mat1 = [];
    for i = 1:100
        %     syn_mat(:,:,:,i) = uint8(syn_mat(:,:,:,i) + mean_img);
        %     syn_mat(:,:,:,i) = single(syn_mat(:,:,:,i));
        temp = syn_mat(:,:,:,i);
        temp = max(-1, min(1,temp)); % [-1,1]
        temp = single( uint8((temp+1)/2 * 255));
        gLow = min(temp(:));
        gHigh = max(temp(:));
        temp = (temp-gLow)/(gHigh-gLow);
        imwrite(temp, 'temp.png');
        syn_mat1 = cat(4, syn_mat1, imread('temp.png'));
       %  syn_mat(:,:,:,i) = temp;
    end
    syn_mats1{c} = syn_mat1;
end
save('./results/coop_500imgs_1000epoch','syn_mats1'); 

a = [];
a{c} = syn_mat1;
class(a{1})

