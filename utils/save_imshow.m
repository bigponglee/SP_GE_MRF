function save_imshow(img, save_name)
    % save_imshow: 保存图像函数
    % Author: Peng Li
    % Date: 2025/04/27
    % 输入参数:
    %   img: 输入图像，[Nx, Ny, 3]
    %   save_name: 保存路径及名称，字符串
    % 输出参数:
    %   无
    % 功能:
    %   保存图像512x512无边框
    h = figure('Visible', 'off', 'Position', [200, 200, 512, 512]);
    imagesc(img);axis image;axis off;
    set(gca, 'XTick', [], 'YTick', [], 'Position', [0 0 1 1]);
    saveas(gcf, save_name, 'jpg')
    close(h);
end