category = 'balloon';
files = dir(['../../image/',category,'/*.JPEG']);

syn_mats1 = [];
count = 0;
for i = 1:10
    syn_mat1 = [];
    for tt = 1:100
        count = count + 1;
        a = imread(['../../image/',category,'/', files(count).name]);
        a = imresize(a, [64 64]);
        if length(size(a)) == 2
            b = syn_mat1(:,:,:,1);
            b(:,:,1) = a;
            b(:,:,2) = a;
            b(:,:,3) = a;
            a = b;
        end
        syn_mat1 = cat(4, syn_mat1, a);
    end
    syn_mats1{i} = syn_mat1;
end

save(['../../image/',category,'_real'],'syn_mats1');