function [X, F] = sample_pareto_set(optima_msg, m)

    interval      = 1 / 10;
    half_interval = interval / 2;
    samples = (half_interval:interval:1-half_interval)';
    k = m - 1;
    opt_mat = double(optima_msg{1}.detach().cpu().numpy());
    grids = [{opt_mat(:,1)}, {opt_mat(:,2)}, repmat({samples},1,k)];
    [grid_arrays{1:2+k}] = ndgrid(grids{:});
    X1 = grid_arrays{1}(:);
    X2 = grid_arrays{2}(:);
    Xs = cell2mat(cellfun(@(g) g(:), grid_arrays(3:end), ...
        'UniformOutput', false));
    X = [Xs, X1, X2];

end