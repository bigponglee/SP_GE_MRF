function superpixel_cluster_idx = superpixel_cluster(superpixels_mean_data, num_clusters, showFlag)
    % superpixel_cluster: 超像素聚类函数
    % Author: Peng Li
    % Date: 2025/04/18
    % 输入参数:
    %   superpixels_mean_data: 超像素均值数据 (N x 3 矩阵)
    %   num_clusters: 期望的聚类数量
    %   showFlag: 是否显示聚类结果 (1: 显示, 0: 不显示)
    % 输出参数:
    %   superpixel_cluster_idx: 超像素聚类的类别索引 (cell 数组)

    if nargin < 3
        showFlag = false; % 默认不显示聚类结果
    end

    %% 执行 K-Means 聚类
    [clusterAssignments, centroids] = kmeans(superpixels_mean_data, num_clusters, ...
                                            'Distance', 'sqeuclidean', ...
                                            'Replicates', 5, ...
                                            'Start', 'plus');
    
    %% 生成超像素聚类的类别索引
    superpixel_cluster_idx = arrayfun(@(k) find(clusterAssignments == k), ...
                                    1:num_clusters, 'UniformOutput', false)';

    %% 可视化聚类结果
    if showFlag
        figure;
        subplot(1, 2, 1);
        scatter(superpixels_mean_data(:, 1), superpixels_mean_data(:, 2), 10, clusterAssignments, 'filled');
        title('K-Means Clustering Result');
        xlabel('T1'); ylabel('T2');
        axis equal;
        grid on;
        subplot(1, 2, 2);
        scatter(superpixels_mean_data(:, 1), superpixels_mean_data(:, 2), 10, clusterAssignments, 'filled');
        hold on;
        plot(centroids(:, 1), centroids(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
        title('K-Means Clustering Result with Centroids');
        xlabel('T1'); ylabel('T2');
        axis equal;
        grid on;
        hold off;
    end
end