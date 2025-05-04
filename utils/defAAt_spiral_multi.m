function [A, At, AtA] = defAAt_spiral_multi(csm_maps)
    % def forward/backward non-Cartesian Fourier undersampling operator for multi coils
    % Author: Peng Li
    % Date: 2025/04/18
    % input: csm_maps - coil sensitivity maps
    % output: A - forward operator
    %         At - backward operator
    %        AtA - Toep AtA operator

    A = @Adef;
    At = @Atdef;
    AtA = @AtAdef;

    function Y_spiral = Adef(X)
        Y_spiral = nufft_multi_op(X, csm_maps);
    end

    function X_down = Atdef(Y_spiral)
        X_down = adjoint_nufft_multi_op(Y_spiral, csm_maps);
    end

    function X_down = AtAdef(X)
        X_down = nufft_ata_multi_op(X, csm_maps);
    end
end
