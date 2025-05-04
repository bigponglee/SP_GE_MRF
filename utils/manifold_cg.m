function PtPxL = manifold_cg(x, Laplasian_cell, P, P_t, res)
    % Author: Peng Li
    % Date: 2025/04/18
    % 函数功能:
    %   求解流形约束项
    % 输入参数:
    %   x: 输入图像，[Nx, Ny, L]
    %   Laplasian_cell: 拉普拉斯矩阵Cell数组，每个元素是一个[num_pixels, num_pixels]的矩阵
    %   P: 前向算子函数
    %   P_t: 后向算子函数
    %   res: 图像大小 [Nx, Ny, L]
    % 输出参数:
    %   PtPxL: 流形约束项，[Nx, Ny, L]

    cell_num = length(Laplasian_cell);

    if ~isequal(size(x), res)
        x = reshape(x, res);
    end
    
    patches = P(x); % patch cell 

    for i = 1:cell_num
        % 计算流形约束项
        patches{i} = Laplasian_cell{i}' * patches{i}; % [pixel_num, L]
    end

    PtPxL = P_t(patches); % [N,N,L]

end
