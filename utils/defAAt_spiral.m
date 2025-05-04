function [A, At, AtA] = defAAt_spiral(m_abs, denorm_flag)
    % def forward/backward non-Cartesian Fourier undersampling operator
    % Author: Peng Li
    % Date: 2025/04/18
    % input: m_abs - normalized weight
    %        denorm_flag - flag for denormalization
    % output: A - forward operator
    %         At - backward operator
    %        AtA - Toep AtA operator
    if nargin < 2
        denorm_flag = true;  % 使用逻辑值更清晰
    end

    A = @Adef;
    At = @Atdef;
    AtA = @AtAdef;

    function Y_spiral = Adef(X)
        Y_spiral = nufft_op(X);
    end

    function X_down = Atdef(Y_spiral)
        X_down = adjoint_nufft_op(Y_spiral);
        if denorm_flag
            X_down = X_down.*m_abs;
        end
    end

    function X_down = AtAdef(X)
        X_down = nufft_ata_op(X);
        if denorm_flag
            X_down = X_down.*m_abs;
        end
    end
end
