function [X_op, break_flag] = optimal_results(X, X_prev, loss, loss_prev, threshold)
    % Author: Peng Li
    % Date: 2025/04/18
    % 函数功能:
    %   执行最优结果的计算，如果loss的变化小于阈值，则停止迭代，否则继续迭代
    % 输入参数:
    %   X: 当前迭代的结果
    %   X_prev: 上一次迭代的结果
    %   loss: 当前迭代的损失
    %   loss_prev: 上一次迭代的损失
    %   threshold: 阈值
    % 输出参数:
    %   X_op: 最优结果
    %   break_flag: 是否停止迭代的标志
    % 示例调用:
    %   [X_op, break_flag] = optimal_results(X, X_prev, loss, loss_prev, threshold);

    break_flag = (loss_prev - loss)/loss_prev <= threshold;

    % 判断loss是否下降，下降时更新X_op
    % 如果loss没有下降，则使用上一次的结果
    if loss_prev > loss
        X_op = X;
    else
        X_op = X_prev;
    end

end
