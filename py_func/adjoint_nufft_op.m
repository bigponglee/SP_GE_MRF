function x = adjoint_nufft_op(k_samples)
    % ADJOINT_NUFFT_OP Adjoint of the NUFFT operator.
    % Author: Peng Li
    % Date: 2025/04/18
    % x = adjoint_nufft_op(k_samples) returns the adjoint of the NUFFT
    % Input:
    % k_samples: k-space samples [L, num_coils, num_samples]
    % Output:
    % x: image [Nx, Ny, L]

    tmp = py.py_func.nufft_for_matlab.nufft_adjoint_op(py.numpy.array(k_samples));
    x = double(tmp{'x'});
end
