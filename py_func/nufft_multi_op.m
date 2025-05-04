function k_samples = nufft_multi_op(im, csm)
    % function k_samples = nufft_multi_op(im, csm)
    % Author: Peng Li
    % Date: 2025/04/18
    % Input:
    %   im: [Nx, Ny, L] image
    %   csm: [1, num_coils, Nx, Ny] coil sensitivity maps
    % Output:
    %   k_samples: [L, num_coils, num_samples] k-space samples

    tmp = py.py_func.nufft_for_matlab_multi.nufft_forward_multi_op( ...
    py.numpy.array(im), py.numpy.array(csm));
    k_samples = double(tmp{'k_samples'});
end
