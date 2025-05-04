function  [X_recon, para_maps_recon, SNR, Loss, process_time] = spge_solver(b, X_gt, Proj_D, A, At, AtA, dic_matching, para)
    % reconstruction using superpixel guided graph embedding
    % Author: Peng Li
    % Date: 2025/04/18
    % 输入参数:
    %   b: 欠采样测量数据，单线圈时为[L, 1, num_samples]，多线圈时为[L, num_coils, num_samples]
    %   X_gt: 真实图像ground truth，[Nx, Ny, L]
    %   Proj_D: 字典投影算子 projection operator, [L, L]
    %   A: 欠采样算子 forward downsampling operator
    %       单线圈时[Nx, Ny, L] -> [L, 1, num_samples]
    %       多线圈时[Nx, Ny, L, num_coils] -> [L, num_coils, num_samples
    %   At: 欠采样算子的转置 adjoint downsampling operator
    %       单线圈时[L, 1, num_samples] -> [Nx, Ny, L]
    %       多线圈时[L, num_coils, num_samples] -> [Nx, Ny, L, num_coils]
    %   AtA: 基于Toep的AtA算子 [Nx, Ny, L] -> [Nx, Ny, L]
    %   dic_matching: 字典匹配算子
    %   para: 超参数结构体
    % 输出参数:
    %   X_recon: 重构图像，[Nx, Ny, L]
    %   para_maps_recon: 重构参数图，[Nx, Ny, L]
    %   SNR: 信噪比
    %   process_time: 处理时间

    %% 初始化
    atb = At(b); % atb [Nx, Ny, L]
    X = atb;
    Y = atb;
    X_recon = atb;
    para_maps = dic_matching(atb); % M0
    % 块提取算子
    [P, Pt] = def_PPt(para_maps, para); % 前向和后向块提取算子
    Laplacian_cell = calc_Laplacian(para_maps, P, para); % 计算拉普拉斯矩阵

    % 预分配内存
    max_iter = para.maxIter;
    Loss = zeros(1, max_iter+1);
    SNR = zeros(1, max_iter+1);
    mu_list = zeros(1, max_iter+1);
    Loss(1) = 1e12;
    if para.print_snr
        SNR(1) = 20 * log10(norm(X_gt(:)) / norm(atb(:) - X_gt(:)));
        fprintf('iteration 0, SNR = %6.4f\n', SNR(1));
    end
    t = para.t;
    acc = para.acc;
    mu = para.mu_min; % 初始步长
    mu_list(1) = mu; % 记录步长

    %% iteration 迭代求解
    tic;
    for iter = 1:max_iter
        % update X_pre
        X_pre = X;
        % gradient descent 梯度下降
        ata_grad = (AtA(Y) - atb);
        % 计算流形约束项
        manifold_grad = para.lambda * manifold_cg(Y, Laplacian_cell, P, Pt, para.res); 
        grad = ata_grad + manifold_grad; % 计算梯度

        % 更新
        X = Y - mu * grad;
        % projection
        X = reshape(X, [], para.res(3)) * Proj_D;
        X = reshape(X, para.res);

        %Accelerated:
        if acc
            % 加速模式
            t_prev = t;
            t = 0.5 * (1 + sqrt(1 + 4 * t ^ 2));
            Y = X + ((t_prev - 1) / t) * (X - X_pre);
        else
            % 非加速模式
            Y = X;
        end

        % calculate loss
        Loss(iter+1) = norm(grad, 'fro')^2; % 计算损失
        if Loss(iter+1) > 1e-8
            ratio = Loss(iter+1) / Loss(iter); % 梯度下降方向是否改变，>1表示改变
            % 根据ratio平滑的调整步长 power decay
            if ratio > 1
                % 发散，降低mu
                mu = max(mu  / (1 - 10 * (1 - ratio)), para.mu_min);
            else
                % 正在收敛，适当增加mu
                mu = min(mu / (1 - 0.1 * (1 - ratio)), para.mu_max);
            end
        end
        mu_list(iter+1) = mu; % 记录步长

        if para.print_snr == 1
            SNR(iter+1) = 20 * log10(norm(X_gt(:)) / norm(X(:) - X_gt(:)));
            fprintf('iter: %d----> loss: %6.4f, SNR: %6.4f, iter step:  %6.4f; \n', iter, Loss(iter+1), SNR(iter+1), mu);
        else
            fprintf('iter: %d----> loss: %6.4f, iter step:  %6.4f; \n', iter, Loss(iter+1), mu);
        end
        [X_recon, break_flag] = optimal_results(X, X_recon, Loss(iter+1), Loss(iter), para.threshold);

        % update Laplacian matrix
        para_maps = dic_matching(X);
        [P, Pt] = def_PPt(para_maps, para); % 前向和后向块提取算子
        Laplacian_cell = calc_Laplacian(para_maps, P, para); % 计算拉普拉斯矩阵
        
        % 收敛处理逻辑
        if break_flag
            if ~acc
                break;  % 非加速模式下直接退出
            else
                t = 1; % 重置加速参数
                % Y = X_recon;
            end
        end
    end
    process_time = toc/60; % 记录处理时间 转换为分钟
    para_maps_recon = dic_matching(X_recon); % 重构参数图
    fprintf('Total time: %f\n', process_time);
    if para.print_snr == 1
        SNR_final = 20 * log10(norm(X_gt(:)) / norm(X_recon(:) - X_gt(:)));
        fprintf('Final SNR: %6.4f\n', SNR_final);
    end
    %% 可视化Loss，SNR，mu曲线
    % 三条曲线在同一张图上
    figure('Name', 'Loss, SNR, mu');
    subplot(3, 1, 1);
    % log scale
    plot(2:iter+1, log10(Loss(2:iter+1)), 'r-', 'LineWidth', 2);
    title('Loss');
    xlabel('Iteration');
    ylabel('Loss（log10）');
    % 设置y轴范围
    ylim([0, max(log10(Loss(2:iter+1)))]);
    grid on;
    subplot(3, 1, 2);
    plot(1:iter+1, SNR(1:iter+1), 'g-', 'LineWidth', 2);
    title('SNR');
    xlabel('Iteration');
    ylabel('SNR (dB)');
    grid on;
    subplot(3, 1, 3);
    plot(1:iter+1, mu_list(1:iter+1), 'b-', 'LineWidth', 2);
    title('mu');
    xlabel('Iteration');
    ylabel('mu');
    grid on;
end