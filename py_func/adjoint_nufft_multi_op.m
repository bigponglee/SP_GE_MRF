function x = adjoint_nufft_multi_op(k_samples, csm)
    % ADJOINT_NUFFT_MULTI_OP Adjoint NUFFT operator for multiple coils
    % Author: Peng Li
    % Date: 2025/04/18
    %  x = adjoint_nufft_multi_op(k_samples, csm)
    % Input:
    %  k_samples: k-space samples [L, 1, num_smaples]
    %  csm: coil sensitivity maps [1, num_coils, Nx, Ny]
    % Output:
    %  x: image [Nx, Ny, L]

    tmp = py.py_func.nufft_for_matlab_multi.nufft_adjoint_multi_op( ...
    py.numpy.array(k_samples), py.numpy.array(csm));

    x = double(tmp{'x'});
end
