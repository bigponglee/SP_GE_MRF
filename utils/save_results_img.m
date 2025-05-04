function save_results_img(map_range, num_maps, sroot, para_gt, sub_title, varargin)
    %% 函数功能：可扩展定量参数图可视化对比
    %% Author: Peng Li
    %% Date: 2025/04/18
    %% 函数输入：
    %% map_range: struct of map_range
    %% num_maps: select map to be shown (2: T1 + T2, 3: T1+T2+PD)
    %% sroot: save root
    %% para_gt: ground truth of parameters
    %% sub_title: cell array of sub_title
    %% varargin: parameters to be compared with ground truth
    %% 函数输出：
    %% 无
    %% 函数示例
    %% save_results_img(map_range, num_maps, sroot, para_gt, {'sub_title1','sub_title2'}, para1, para2);

    if length(sub_title) ~= length(varargin)
        error('sub_title and varargin must have the same length');
    end

    para_num = length(varargin);
    titles = {'T1', 'T2', 'PD'};
    ranges = [map_range.T1_value, map_range.T2_value, map_range.PD_value];
    error_ranges = [map_range.T1_value_error, map_range.T2_value_error, map_range.PD_value_error];

    maps_name = hot;
    for k = 1:num_maps
        save_fig(para_gt(:, :, k), [0 ranges(k)], [sroot, titles{k}, '_gt'], maps_name);
    end

    for i = 1:para_num
        for k = 1:num_maps
            save_fig(varargin{i}(:, :, k), [0 ranges(k)], [sroot, titles{k}, '_', sub_title{i}], maps_name);
            save_fig(abs(varargin{i}(:, :, k) - para_gt(:, :, k)), [0, error_ranges(k)], [sroot, titles{k}, '_', sub_title{i}, '_error'], maps_name);
        end
    end

end

function save_fig(x, show_range, save_name, maps_name)
    % save_fig(x, show_range, save_name, maps_name)
    % save_fig: save the figure in jpg format
    % x: the matrix to be shown
    % show_range: the range of the matrix to be shown
    % save_name: the name of the saved figure
    % maps_name: the name of the colormap
    %NMSE
    h = figure('Visible', 'off', 'Position', [200, 200, 512, 512]);
    imagesc(x, show_range); axis image; colormap(maps_name);
    set(gca, 'XTick', [], 'YTick', [], 'Position', [0 0 1 1]);
    saveas(gcf, save_name, 'jpg')
    close(h);
end