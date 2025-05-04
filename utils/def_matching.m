function [dic_matching, Proj_D] = def_matching(D, LUT, matching_var, matching_batch)
    % function dic_matching = def_matching(D, LUT, matching_var, matching_batch)
    % Author: Peng Li
    % Date: 2025/04/18
    % define the matching function
    % input
    %   D: dictionary [para_num, L]
    %   LUT: look up table [para_num, 2]
    %   matching_var: matching threshold
    %   matching_batch: batch size for matching
    %   PD_norm: PD normlized factor
    % output
    %   dic_matching: matching function
    %   Proj_D: projection operator L*L

    dic_matching = @matching_def;

    %% 投影算子 projection operator
    base = orth(D.'); % L*r
    base = base.'; % r*L
    pinv_D = pinv(base); %Moore-Penrose Pseudoinverse 逆 L*r
    Proj_D = pinv_D * base; %projection operator L*L

    function para_maps_recon = matching_def(X_recon)
        % function para_maps_recon = matching_def(X_recon)
        % matching function
        % input
        %   X_recon: reconstructed images [N, N, L]
        % output
        %   para_maps_recon: reconstructed parameter maps [N, N, 3]
        tmp = py.py_func.dic_matching.build_maps_mat(...
                        py.numpy.array(X_recon), ...
                        py.numpy.array(D), ...
                        py.numpy.array(LUT), ...
                        matching_var, matching_batch);
        para_maps_recon = cat(3, ...
                        double(tmp{'t1'}), ...
                        double(tmp{'t2'}), ...
                        double(tmp{'m0'}));
    end
end
