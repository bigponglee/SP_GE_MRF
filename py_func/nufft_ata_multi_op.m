function x_ata = nufft_ata_multi_op(im, csm)
    % function x_ata = nufft_ata_multi_op(im, csm)
    % Author: Peng Li
    % Date: 2025/04/18
    % Input:
    % im: [Nx, Ny, L] image
    % csm: [1, ncoils Nx, Ny] coil sensitivity maps
    % Output:
    % x_ata: [Nx, Ny, L] image

    tmp = py.py_func.nufft_for_matlab_multi.nufft_toep_ata_multi_op(py.numpy.array(im), py.numpy.array(csm));
    x_ata = double(tmp{'x_ata'});
end
