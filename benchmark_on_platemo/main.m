%% ============================================================
%  Python Environment Setup
%% ============================================================
PYTHON_PATH = '.\.venv\Scripts\python.exe';

% Resolve to absolute, canonical path for reliable comparison
PYTHON_PATH_ABS = canonical_path(PYTHON_PATH);

pe = pyenv;
if pe.Status == "NotLoaded"
    pe = pyenv('Version', PYTHON_PATH_ABS, 'ExecutionMode', 'InProcess');
else
    current_abs = canonical_path(char(pe.Executable));
    if ~strcmpi(current_abs, PYTHON_PATH_ABS)
        error(['Python environment mismatch:\n' ...
               '  Current   : %s\n' ...
               '  Requested : %s\n' ...
               'Restart MATLAB to switch.'], current_abs, PYTHON_PATH_ABS);
    end
end
fprintf('CurrentPython: %s\n', char(pe.Executable));


function p = canonical_path(p)
% Resolve to absolute path; fall back to the input if the file is missing.
    info = dir(p);
    if ~isempty(info)
        p = fullfile(info(1).folder, info(1).name);
    else
        % File not found: at least make it absolute relative to pwd
        if ~startsWith(p, ["\\", "/"]) && isempty(regexp(p, '^[A-Za-z]:', 'once'))
            p = fullfile(pwd, p);
        end
    end
end
%% ============================================================
%  Python Module Path & Imports
%% ============================================================
SCRIPT_PATH = '.\x_msg';
if count(py.sys.path, SCRIPT_PATH) == 0
    insert(py.sys.path, int32(0), SCRIPT_PATH);
end

% Project modules (reloaded so edits take effect)
construct_msg = reload_py('construct_msg_landscape');
sampling_py   = reload_py('sampling');
make_mo_msg   = reload_py('make_multi_objective_msg');
extract_feat  = reload_py('extract_features');
torch_py      = py.importlib.import_module('torch');

%% ============================================================
%  Experiment Configuration
%% ============================================================

BASE_SEED     = 42;
D_class       = [2, 5, 10];

for d = 1:length(D_class)
    D = D_class(d);
    fprintf('=== Starting experiments for D=%d ===\n', D);
    NUM_GAUSSIANS = 50 * D;
    DEVICE        = 'cuda';
    NUM_TRIALS    = 31;
    NUM_SAMPLES   = 500 * D;

    types = ["max_max_max", "max_max_min", "max_min_max", "max_min_min", ...
            "min_max_max", "min_max_min", "min_min_max", "min_min_min"];

    for i = 1:length(types)
        fprintf('Type %d: %s\n', i, types(i));

        type = types(i); 

        CKPT_PATH = ".\res_rq2\msg_ela\results_" + type + "_" + num2str(D) + "d.pt";
        CSV_FILE  = "IGDX_scores_" + type + "_" + num2str(D) + "d.csv";
        FEATURE_LIST = py.list({'optima_feature','fdc_feature', ...
                                'disp_feature','r2_feature'});
        P_LIST       = py.list({0.1});

        %% ============================================================
        %  Sobol Gaussian Means & theta History
        %% ============================================================
        means = sampling_py.sobol_sampling( ...
            py.int(D), py.int(NUM_GAUSSIANS), ...
            pyargs('device', py.str(DEVICE), 'seed', py.int(BASE_SEED)));
        means = means.to(py.str(DEVICE));

        ckpt          = torch_py.load(CKPT_PATH);
        theta_history = double(ckpt{'theta_history'}.detach().cpu().numpy());

        %% ============================================================
        %  Resume: Load completed (idx, seed) pairs from CSV
        %% ============================================================
        completed = containers.Map('KeyType', 'char', 'ValueType', 'logical');
        if isfile(CSV_FILE)
            T = readtable(CSV_FILE);
            bad = any(ismissing(T(:, {'idx','seed'})), 2);
            if any(bad)
                warning('Dropping %d corrupted row(s) from resume map', sum(bad));
                T(bad, :) = [];
            end
            for r = 1:height(T)
                k = sprintf('%d_%d', T.idx(r), T.seed(r));
                completed(k) = true;
            end
            fprintf('Resume mode: %d rows already in %s\n', height(T), CSV_FILE);
        else
            fprintf('Fresh run: %s not found\n', CSV_FILE);
        end

        make_key = @(i, s) sprintf('%d_%d', i, s);

        %% ============================================================
        %  Main Loop: evaluate HREA (IGDX) for each theta x seed x eta
        %% ============================================================
        for idx = 1:size(theta_history, 1)

            % Skip this idx entirely if every (idx, seed) is done
            all_done = true;
            for t = 1:NUM_TRIALS
                if ~isKey(completed, make_key(idx, BASE_SEED + t))
                    all_done = false; break;
                end
            end
            if all_done
                fprintf('Skip idx=%d (all %d trials done)\n', idx, NUM_TRIALS);
                continue;
            end

            % --- Build MSG landscape for this theta ---
            theta = py.torch.tensor(theta_history(idx, :), pyargs( ...
                'dtype', py.torch.float32, 'device', py.str(DEVICE)));
            msg   = construct_msg.MSGLandscape(means, theta);

            % --- ELA features ---
            features = extract_feat.compute_features( ...
                py.int(NUM_SAMPLES), theta, means, P_LIST, DEVICE, FEATURE_LIST);
            num_local_optima = double(features{'num_local_optima'}.cpu().numpy());
            fdc              = double(features{'fdc'}.cpu().numpy());
            dispersion       = double(features{'disp_10pct'}.cpu().numpy());

            % --- Multi-objective problem built on MSG ---
            mo_msg = make_mo_msg.make_multi_objective_msg(pyargs( ...
                'm', py.int(2), 'dim_msg', py.int(D), ...
                'function_g', msg, 'pf_shape', py.str('convex')));

            multi_msg = @(x) msg_wrapper_batch(x, mo_msg, DEVICE);
            PRO       = MyProblem(mo_msg, DEVICE, D+1);

            % --- Reference sets: local / global Pareto sets ---
            optima     = msg.find_optima_exact(py.float(0.0));
            local_opt  = optima(1);   % local optima
            global_opt = optima(3);   % global optima

            X_local  = sample_pareto_set(local_opt,  2);
            X_global = sample_pareto_set(global_opt, 2);
            F_local  = multi_msg(X_local);
            F_global = multi_msg(X_global);

            % Epsilon-local optima in decision space (eps = 0.05)
            [pop_filtered, ~] = get_epsilon_local_optima( ...
                X_local, F_local, F_global, 0.05);
            n_eps_local = size(pop_filtered, 1);

            % --- Run HREA across seeds and eta values ---
            for trial = 1:NUM_TRIALS
                seed = BASE_SEED + trial;
                if isKey(completed, make_key(idx, seed))
                    continue;
                end

                rng(seed);
                py.random.seed(seed);

                scores_IGDX = zeros(1, 4);
                scores_Global_IGDX = zeros(1, 4);
                scores_HV   = zeros(1, 4);
                HREA     = custom_HREA();
                HREA.Solve(PRO);

                archive = HREA.Archive;
                final_pop_HREA = HREA.result{end};
                scores_IGDX(1)    = IGDX(archive, pop_filtered);
                scores_Global_IGDX(1) = IGDX(archive, X_global);
                scores_HV(1)      = HV(final_pop_HREA, F_global);
                fprintf('HREA trial %d/%d: IGDX=%.4f, IGDX_Global=%.4f, HV=%.4f\n', trial, NUM_TRIALS, scores_IGDX(1), scores_Global_IGDX(1), scores_HV(1));

                MMEAWI    = custom_MMEAWI();
                MMEAWI.Solve(PRO);
                archive = MMEAWI.Archive;
                final_pop_MMEAWI = MMEAWI.result{end};
                scores_IGDX(2)    = IGDX(archive, pop_filtered);
                scores_Global_IGDX(2) = IGDX(archive, X_global);
                scores_HV(2)      = HV(final_pop_MMEAWI, F_global);
                fprintf('MMEAWI trial %d/%d: IGDX=%.4f, IGDX_Global=%.4f, HV=%.4f\n', trial, NUM_TRIALS, scores_IGDX(2), scores_Global_IGDX(2), scores_HV(2));

                NSGAII     = custom_NSGAII();
                NSGAII.Solve(PRO);
                final_pop_NSGAII = NSGAII.result{end};
                scores_IGDX(3)    = IGDX(final_pop_NSGAII, pop_filtered);
                scores_Global_IGDX(3) = IGDX(final_pop_NSGAII, X_global);
                scores_HV(3)      = HV(final_pop_NSGAII, F_global);
                fprintf('NSGAII trial %d/%d: IGDX=%.4f, IGDX_Global=%.4f, HV=%.4f\n', trial, NUM_TRIALS, scores_IGDX(3), scores_Global_IGDX(3), scores_HV(3));

                CPDEA     = custom_CPDEA();
                CPDEA.Solve(PRO);
                archive = CPDEA.Archive;
                final_pop_CPDEA = CPDEA.result{end};
                scores_IGDX(4)    = IGDX(archive, pop_filtered);
                scores_Global_IGDX(4) = IGDX(archive, X_global);
                scores_HV(4)      = HV(final_pop_CPDEA, F_global);
                fprintf('CPDEA trial %d/%d: IGDX=%.4f, IGDX_Global=%.4f, HV=%.4f\n', trial, NUM_TRIALS, scores_IGDX(4), scores_Global_IGDX(4), scores_HV(4));


                fprintf('idx=%d, trial=%d/%d finished\n', idx, trial, NUM_TRIALS);

                append_row(CSV_FILE, idx, seed, n_eps_local, ...
                        num_local_optima, fdc, dispersion, scores_IGDX, scores_Global_IGDX, scores_HV);

                completed(make_key(idx, seed)) = true;
            end
        end
    end
end

%% ============================================================
%  Local Helper Functions
%% ============================================================
function m = reload_py(name)
    % Import and force-reload a Python module.
    m = py.importlib.import_module(name);
    py.importlib.reload(m);
end

function append_row(csv_file, idx, seed, n_eps, n_local, fdc, disp10, scores_IGDX, scores_Global_IGDX, scores_HV)
    % Append a result row to CSV; write header on first call.
    if ~isfile(csv_file)
        header = {'idx','seed','num_epsilon_local_optima', ...
                  'num_local_optima','fdc','disp_10pct', ...
                  'IGDX_HREA','IGDX_MMEAWI','IGDX_NSGAII','IGDX_CPDEA', ...
                  'IGDX_Global_HREA','IGDX_Global_MMEAWI','IGDX_Global_NSGAII','IGDX_Global_CPDEA', ...
                  'HV_HREA','HV_MMEAWI','HV_NSGAII','HV_CPDEA'};
        writecell(header, csv_file);
    end
    row = [{idx}, {seed}, {n_eps}, {n_local}, {fdc}, {disp10}, num2cell(scores_IGDX), num2cell(scores_Global_IGDX), num2cell(scores_HV)];
    writecell(row, csv_file, 'WriteMode', 'append');
end