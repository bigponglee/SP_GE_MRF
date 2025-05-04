function [P, Pt] = def_PPt(para_maps, para)
    % def forward/backward patch operator
    % Author: Peng Li
    % Date: 2025/04/18
    % 函数功能:
    %   执行超像素分割和聚类，生成前向和后向块提取算子
    % 输入参数:
    %   para_maps: 超参数图，[Nx, Ny, L]
    %   para: 超参数结构体
    %   涉及到的超参数:
    %       para.spBase: 超像素分割的基础图像 (1: T1, 2: T2, 3: T1+T2+PD, 12: T1+T2)
    %       para.numSuperpixels: 期望的超像素数量
    %       para.compactness: 超像素分割的紧凑度
    %       para.num_clusters: 超像素聚类的数量
    %       para.res: 图像大小 [Nx, Ny, L]
    % 输出参数:
    %   P: 前向算子函数
    %   Pt: 后向算子函数

    % 1. 超像素分割
    [superpixelIdxList, superpixel_mean_param] = superpixel_gen(para_maps, para.spBase, para.numSuperpixels, para.compactness, para.showFlag);
    % 2. 超像素聚类
    superpixel_cluster_idx = superpixel_cluster(superpixel_mean_param, para.num_clusters, para.showFlag);

    num_cols = para.res(1);
    num_rows = para.res(2);
    num_pixels = num_cols * num_rows;

    % 3. 预计算所有超像素索引的展开形式
    superpixel_indices = cellfun(@(x) cell2mat(superpixelIdxList(x)), ...
                            superpixel_cluster_idx, 'UniformOutput', false);

    % 4. Define the forward operator and its adjoint
    P = @patch_image;
    Pt = @unpatch_image;

    function patches = patch_image(x)
        %% patch_image: 前向算子
        % 输入参数:
        %   x: 输入图像，[Nx, Ny, L]
        % 输出参数:
        %   patches: 超像素块，Cell数组，每个元素是一个[num_pixels, L]的矩阵
        num_channels = size(x, 3); % X时是L，para maps时是3（参数数目）
        % 从数据X中抽取超像素块，构成patch
        x_reshaped = reshape(x, num_pixels, num_channels);     
        patches = cellfun(@(idx) x_reshaped(idx,:), ...
                        superpixel_indices, 'UniformOutput', false);
    end

    function x = unpatch_image(patches)
        %% unpatch_image: 后向算子
        % 输入参数:
        %   patches: 超像素块，Cell数组，每个元素是一个[num_pixels, L]的矩阵
        % 输出参数:
        %   x: 重构图像，[Nx, Ny, L]
        num_channels = size(patches{1}, 2);
        % 将patch中的像素值重新填充到原始图像中
        x = zeros(num_pixels, num_channels);
        for k = 1:length(superpixel_cluster_idx)
            x(superpixel_indices{k}, :) = patches{k};
        end
        x = reshape(x, num_cols, num_rows, num_channels);
    end
end
