function [superpixelIdxList, superpixel_mean_param] = superpixel_gen(para_map, spBase, numSuperpixels, compactness, showFlag)
    % superpixel_gen - 超像素分割自定义封装函数
    % Author: Peng Li
    % Date: 2025/04/18
    % Arguments:
    % para_map - 输入图像，此处为参数图像 (T1, T2, PD)
    % spBase - 选择的基础图像 (1: T1, 2: T2, 3: T1+T2+PD, 12: T1+T2)
    % numSuperpixels - 期望的超像素数量
    % compactness - 超像素的紧凑性参数 (越大越紧凑)
    % showFlag - 是否显示分割结果 (1: 显示, 0: 不显示)
    % Returns:
    % superpixelIdxList - 超像素块的线性索引列表 (cell 数组)
    % superpixel_mean_param - 超像素平均参数值差距矩阵
    % N - 实际生成的超像素数量

    if nargin < 5
        showFlag = false; % 默认不显示聚类结果
    end

    % 预处理函数封装
    function img = preprocess_channel(channel, max_val)
        channel(channel > max_val) = max_val;
        channel = channel ./ max(channel(:));
        img = im2uint8(channel);
    end

    %% 超像素分割的图像选择与预处理
    switch spBase
        case 1
            img = preprocess_channel(squeeze(para_map(:,:,1)), 3000);
            sp_mode = 'T1';
        case 2
            img = preprocess_channel(squeeze(para_map(:,:,2)), 1000);
            sp_mode = 'T2';
        case 3
            T1 = preprocess_channel(para_map(:,:,1), 3000);
            T2 = preprocess_channel(para_map(:,:,2), 1000);
            PD = para_map(:,:,3) ./ max(para_map(:,:,3), [], 'all');
            img = im2uint8(cat(3, T1, T2, PD));
            sp_mode = 'T1+T2+PD';
        case 12
            T1 = preprocess_channel(para_map(:,:,1), 3000);
            T2 = preprocess_channel(para_map(:,:,2), 1000);
            img = im2uint8(cat(3, T1, T2, T1));
            sp_mode = 'T1+T2';
        otherwise
            error('Invalid spBase value. It should be 1, 2, 3, or 12.');
    end

    para_map_flat = reshape(para_map, [], 3); % 将参数图像转换为二维矩阵 (每行是一个像素的 RGB 值)

    %% 执行超像素分割 (使用 Simple Linear Iterative Clustering (SLIC) 算法)
    % L: 标签矩阵，每个像素的值是其所属超像素的索引 (1 到 N)
    % N: 实际生成的超像素数量
    [L, N] = superpixels(img, numSuperpixels, 'Compactness', compactness, 'Method', 'slic', 'NumIterations', 50);
    fprintf('期望超像素数量: %d, 实际生成数量: %d\n', numSuperpixels, N);

    %% 由超像素块的坐标生成超像素块的索引，并生成超像素平均参数值差距矩阵
    superpixelIdxList = arrayfun(@(k) find(L == k), 1:N, 'UniformOutput', false)';
    % 使用accumarray计算均值
    pixel_labels = double(L(:));
    superpixel_mean_param = zeros(N, 3);
    for ch = 1:3
        superpixel_mean_param(:,ch) = accumarray(pixel_labels, para_map_flat(:,ch), [N 1], @mean);
    end

    % 可视化超像素分割结果
    %% 可视化优化
    if showFlag
        % 预计算颜色映射
        cmap = lines(N);
        colored_sp_img = reshape(cmap(pixel_labels,:), [size(L), 3]);
        
        % 混合图像
        alpha = 0.3;
        img_display = im2double(squeeze(img(:,:,1)));
        blended_img = alpha * colored_sp_img + (1 - alpha) * repmat(img_display, 1, 1, 3);
        
        figure('Name', ['Superpixel Segmentation Result, SP mode: ', sp_mode], 'NumberTitle', 'off');
        subplot(1,2,1); imshow(blended_img); title('超像素分割叠加结果');
        subplot(1,2,2); imshow(colored_sp_img); title('超像素分割结果');
        % save_imshow(blended_img, 'results//500_vds_noisy//sp//sp_blended_0');
        % save_imshow(colored_sp_img,'results//500_vds_noisy//sp//sp_colored_0');
    end
end