function [  ] = interpolator( config, net,  syn_mat_all, epoch )
% syn_mats is a 1*num_batch cell
% Here, we consider three modes: line, sphere or both

%% put the cell into a big mat
% n_batch = size(syn_mats, 2);
% syn_mat_all = [];
% batch_end = 0;
% for i_batch = 1:n_batch
%     batch_size = size(syn_mats{i_batch}, 4); % the 4th dimension is the batchsize
%     batch_start = batch_end + 1;
%     batch_end = batch_start + (batch_size-1);
%     syn_mat_all(:,:,:, batch_start:batch_end) = syn_mats{i_batch};
% end

num_Z = size(syn_mat_all, 4);
%% do the interpolation to get grid of Z value. 
% for line interpolation, choose config.n_pairs, for each pair, by default,
% we get 8 equally spaced values
% for sphere interpolation, choose 4 example Z, then use sin, cos to
% parametriz them, for one parameter, by default, get 8 equally spaced
% values, for the other, get config.n_parsamp values. 

switch config.interp_type
    case 'line'
        interp_Z = zeros(config.n_pairs * 8, config.z_sz, 'single');
        line_points = linspace(0, 1, 8);
        for i_pair = 1:config.n_pairs
            % choose two points
            current_pair = randperm(num_Z, 2);
            %example_Z = syn_mat_all(:,:,:,current_pair);
            left_Z = syn_mat_all(:,:,:,current_pair(1));
            right_Z = syn_mat_all(:,:,:, current_pair(2));
            % interp_z is 8 * z_dim
            i_interp_Z = (line_points')*left_Z + (1.0-line_points')*right_Z;
            interp_Z((i_pair-1)*8 + 1: i_pair*8, :) = i_interp_Z;
        end
        draw_interpolation_figures(config, net, interp_Z, 'line',epoch);
    case 'sphere'
        phi_points = linspace(0, pi/2, 8);
        theta_points = linspace(0, pi/2, config.n_parsamp);
        interp_Z = zeros(8* config.n_parsamp, config.z_sz, 'single');
        %choose 4 example Z points
        four_example = randperm(num_Z, 4);
        first_Z = syn_mat_all(:,:,:,four_example(1));
        second_Z = syn_mat_all(:,:,:,four_example(2));
        third_Z = syn_mat_all(:,:,:,four_example(3));
        four_Z = syn_mat_all(:,:,:,four_example(4));
        for i_theta = 1:config.n_parsamp
            i_interp_Z = cos(theta_points(i_theta))*(cos(phi_points)') * first_Z ...
                + cos(theta_points(i_theta)) * (sin(phi_points)')*second_Z ...
                + sin(theta_points(i_theta)) * (cos(phi_points)') * third_Z ...
                + sin(theta_points(i_theta)) * (sin(phi_points)') * four_Z;
            interp_Z((i_theta-1)*8 + 1 : i_theta * 8, :) = i_interp_Z;
        end
        draw_interpolation_figures(config, net, interp_Z, 'sphere', epoch);
    case 'both'
        interp_Z_line = zeros([config.n_pairs * 8, config.z_sz(end)], 'single');
        line_points = linspace(0, 1, 8);
        for i_pair = 1:config.n_pairs
            % choose two points
            current_pair = randperm(num_Z, 2);
            %example_Z = syn_mat_all(:,:,:,current_pair);
            left_Z = syn_mat_all(:,:,:,current_pair(1));
            left_Z = reshape(left_Z, 1, size(left_Z, 3));
            right_Z = syn_mat_all(:,:,:, current_pair(2));
            right_Z = reshape(right_Z, 1, size(right_Z, 3));
            % interp_z is 8 * z_dim
            i_interp_Z = (line_points')*left_Z + (1.0-line_points')*right_Z;
            interp_Z_line((i_pair-1)*8 + 1: i_pair*8, :) = i_interp_Z;
        end
        draw_interpolation_figures(config, net, interp_Z_line, 'line',epoch);
        
        % do sphere
        phi_points = linspace(0, pi/2, 8);
        theta_points = linspace(0, pi/2, config.n_parsamp);
        interp_Z_S = zeros([8* config.n_parsamp, config.z_sz(end)], 'single');
        %choose 4 example Z points
        four_example = randperm(num_Z, 4);
        first_Z = syn_mat_all(:,:,:,four_example(1));
        second_Z = syn_mat_all(:,:,:,four_example(2));
        third_Z = syn_mat_all(:,:,:,four_example(3));
        four_Z = syn_mat_all(:,:,:,four_example(4));
        
        first_Z = reshape(first_Z, 1, size(first_Z, 3));
        second_Z = reshape(second_Z, 1, size(second_Z, 3));
        third_Z = reshape(third_Z, 1, size(third_Z, 3));
        four_Z = reshape(four_Z, 1, size(four_Z, 3));
        for i_theta = 1:config.n_parsamp
            i_interp_Z = cos(theta_points(i_theta))*(cos(phi_points)') * first_Z ...
                + cos(theta_points(i_theta)) * (sin(phi_points)')*second_Z ...
                + sin(theta_points(i_theta)) * (cos(phi_points)') * third_Z ...
                + sin(theta_points(i_theta)) * (sin(phi_points)') * four_Z;
            interp_Z_S((i_theta-1)*8 + 1 : i_theta * 8, :) = i_interp_Z;
        end
        draw_interpolation_figures(config, net, interp_Z_S, 'sphere', epoch);
        
end

end

