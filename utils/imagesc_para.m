function imagesc_para(map_range, num_maps, para_gt, sub_title, varargin)
    %% 函数功能：可扩展定量参数图可视化对比
    %% Author: Peng Li
    %% Date: 2025/04/18
    %% 函数输入：
    %% map_range: struct of map_range
    %% num_maps: select map to be shown (2: T1 + T2, 3: T1+T2+PD)
    %% para_gt: ground truth of parameters
    %% sub_title: cell array of sub_title
    %% varargin: parameters to be compared with ground truth
    %% 函数输出：
    %% 无
    %% 函数示例：
    %% imagesc_para(map_range, para_gt, {'sub_title1','sub_title2'}, para1, para2);
    %% imagesc_para(map_range, para_gt, {'sub_title1','sub_title2','sub_title3'}, para1, para2, para3);

    if length(sub_title) ~= length(varargin)
        error('sub_title and varargin must have the same length');
    end

    para_num = length(varargin) + 1;
    titles = {'T1', 'T2', 'PD'};
    ranges = [map_range.T1_value, map_range.T2_value, map_range.PD_value];
    error_ranges = [map_range.T1_value_error, map_range.T2_value_error, map_range.PD_value_error];
    
    %% para maps show
    figure('Name', 'T1 T2 PD images', 'NumberTitle', 'off');
    colormap(hot);
    
    % 显示GT图像
    for k = 1:num_maps
        subplot(num_maps, para_num, (k-1)*para_num + 1);
        imagesc(para_gt(:, :, k), [0, ranges(k)]);
        title([titles{k} ' GT'], 'FontSize', 12.5);
        axis off; axis image; colorbar;
    end
    
    % 显示比较图像
    for i = 1:para_num-1
        for k = 1:num_maps
            subplot(num_maps, para_num, (k-1)*para_num + i + 1);
            imagesc(varargin{i}(:, :, k), [0, ranges(k)]);
            title([titles{k} ' ' sub_title{i}], 'FontSize', 12.5);
            axis off; axis image; colorbar;
        end
    end

    %% error maps show
    figure('Name', 'Error maps', 'NumberTitle', 'off');
    colormap(hot);
    y_pos = size(para_gt, 1) + 12;
    
    for i = 1:para_num-1
        for k = 1:num_maps
            nmse = goodnessOfFit(reshape(varargin{i}(:, :, k), [], 1), ...
                                reshape(para_gt(:, :, k), [], 1), 'NMSE');
            
            subplot(num_maps, para_num-1, (k-1)*(para_num-1) + i);
            imagesc(abs(varargin{i}(:, :, k) - para_gt(:, :, k)), [0, error_ranges(k)]);
            axis off; axis image; colorbar;
            title([titles{k} ' ' sub_title{i}], 'FontSize', 20);
            text(20, y_pos, ['NMSE=', num2str(nmse)], 'FontSize', 12.5);
        end
    end
end
