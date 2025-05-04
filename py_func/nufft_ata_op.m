function x_ata = nufft_ata_op(im)
    % Author: Peng Li
    % Date: 2025/04/18
    % x_ata = nufft_ata_op(im)
    % Input:
    % im - image [Nx, Ny, L]
    % Output:
    % x_ata - [Nx, Ny, L]

    tmp = py.py_func.nufft_for_matlab.nufft_toep_ata_op(py.numpy.array(im));
    x_ata = double(tmp{'x_ata'});
end
