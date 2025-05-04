function super_pixel_cluster_show(img, superpixelIdxList, superpixel_cluster_idx, save_index)
    % 超像素及其聚类结果可视化与保存
    % 输入参数:
    %   img: 原始图像 (Nx x Ny) 
    %   superpixelIdxList: 超像素块的线性索引列表 (cell 数组)
    %   superpixel_cluster_idx: 超像素聚类的类别索引 (cell 数组)
    %   save_index: 保存索引
    % 输出参数:
    %   无
    % 功能:
    %   1. 显示超像素分割结果
    %   2. 显示超像素聚类结果：聚为一类的超像素块用相同颜色表示
    %   3. 保存超像素分割结果图和超像素聚类结果图

    num_clusters = length(superpixel_cluster_idx); % 聚类数量
    % 为每类超像素分配不同的颜色
    cmap = lines(num_clusters); % 使用 lines 颜色映射
    % 显示超像素分割结果
    % 一个子图显示原始图像与超像素分割结果的叠加
    % 另一个子图显示超像素聚类结果
    img_display = im2double(squeeze(img(:,:,1)));
    img_display = repmat(img_display, [1, 1, 3]); % 扩展为 RGB 图像
    img_display = img_display ./ max(img_display(:)); 
    % 混合图像
    alpha = 0.3;
    colored_sp_img = zeros(size(img_display)); % 初始化混合图像
    colored_sp_img = reshape(colored_sp_img, [], 3); % 转换为二维矩阵

    for k = 1:num_clusters
        % 为第 k 类超像素块分配颜色
        color = cmap(k, :); % 获取颜色映射中的颜色
        % 绘制超像素块边界
        for i = 1:length(superpixel_cluster_idx{k})
            % 获取第 k 类超像素块的索引
            idx = superpixel_cluster_idx{k}(i);
            % 获取第 k 类超像素块的像素索引
            pixel_idx = superpixelIdxList{idx}; % 超像素块的像素索引, 绝对索引
            % 将第 k 类超像素块的像素值设置为颜色
            colored_sp_img(pixel_idx, :) = repmat(color, [length(pixel_idx), 1]);
        end
    end
    colored_sp_img = reshape(colored_sp_img, size(img_display)); % 转换为三维矩阵
    % 混合图像
    blended_img = alpha * img_display + (1 - alpha) * colored_sp_img; % 混合图像
    % 显示混合图像
    % figure('Name', 'Superpixel Segmentation and Clustering Result (Blended)')
    % imshow(blended_img);
    % figure('Name', 'Superpixel Segmentation and Clustering Result (Colored)')
    % imshow(colored_sp_img);

    % 保存超像素分割结果图和超像素聚类结果图
    save_imshow(blended_img, ['results//500_vds_noisy//sp//sp_blended_cluster_', num2str(save_index)]);
    save_imshow(colored_sp_img,['results//500_vds_noisy//sp//sp_colored_cluster_', num2str(save_index)]);
end
    



