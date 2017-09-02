function [] = draw_figures(config, syn_mat, iter, mean_img, SSD, layer, prefix)

if nargin < 7
    prefix = '';
else
    prefix = [prefix, '_'];
end

[I_syn, syn_mat_norm] = convert_syns_mat(config, mean_img, syn_mat);

for i = 1:size(syn_mat_norm, 4)
    imwrite(syn_mat_norm(:,:,:,i), [config.figure_folder, prefix, num2str(layer, 'layer_%02d_'), num2str(i, '%03d.png')]);
end

imwrite(I_syn, [config.Synfolder, prefix, num2str(layer, 'layer_%02d_'), num2str(iter, 'dense_original_%04d'), '.png']);


if ~isempty(SSD)
    plot(1:iter, SSD(1:iter), 'r', 'LineWidth', 3);
    axis([min(iter, 1), iter+1, 0,  max(SSD(min(iter, 1):end)) * 1.2]);
    title('Matched statistics')
    drawnow;
end

end