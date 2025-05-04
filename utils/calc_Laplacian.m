function Laplacian_cell = calc_Laplacian(para_maps, P, para)
    % 依据参数图，根据超像素块，计算每个超像素聚类的拉普拉斯矩阵
    % Author: Peng Li
    % Date: 2025/04/18
    % 输入参数:
    %   para_maps: 超参数图，[Nx, Ny, L]
    %   P: 前向块提取算子
    %   para: 超参数结构体\
    %   涉及到的超参数:
    %       para.sig: 高斯核的标准差
    % 输出参数:
    %   Laplacian_cell: 由每个超像素聚类的拉普拉斯矩阵组成的Cell数组

    % 超像素块提取
    patches = P(para_maps); % 前向块提取
    num_patches = length(patches); % 超像素块数量
    Laplacian_cell = cell(num_patches, 1); % 初始化拉普拉斯矩阵Cell数组

    % 超参数
    sig_sq = para.sig;

    % 计算每个超像素聚类的拉普拉斯矩阵
    for k = 1:num_patches
        % 计算每个超像素块的拉普拉斯矩阵
        patch = patches{k};
        % 1. 计算每个像素点之间的欧式距离
        dist2 = pdist2(patch, patch);
        % 2. 计算高斯核
        W = exp(-dist2/(sig_sq+ 1e-6));
        % 3. 计算归一化拉普拉斯矩阵
        D = sum(W, 2); % D是对角矩阵
        D_inv = 1 ./ sqrt(D); % D_inv是D的逆平方根
        
        % 方案一 使用稀疏对角矩阵
        D_inv = spdiags(D_inv, 0, length(D), length(D));  % 创建稀疏对角矩阵
        D = spdiags(D, 0, length(D), length(D));
        Laplacian_cell{k} = D_inv * (D - W) * D_inv; % 归一化拉普拉斯矩阵L = D^(-1/2) * (D - W) * D^(-1/2)
        
        % 方案二利用广播特性直接计算归一化拉普拉斯矩阵
        % W_norm = W .* (D_inv * D_inv');  % 等价于 D^(-1/2) * W * D^(-1/2)
        % Laplacian_cell{k} = speye(size(W)) - W_norm;  % L = I - D^(-1/2)*W*D^(-1/2)

    end
end


