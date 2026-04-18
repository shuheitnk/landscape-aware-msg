%% ============================================================
%  Python Environment Setup
%% ============================================================
PYTHON_PATH = '.\.venv\Scripts\python.exe';
pe = pyenv;
if pe.Status == "NotLoaded"
    pe = pyenv('Version', PYTHON_PATH, 'ExecutionMode', 'InProcess');
elseif ~strcmpi(pe.Executable, PYTHON_PATH)
    error(['Python environment mismatch:\n' ...
           '  Current   : %s\n' ...
           '  Requested : %s\n' ...
           'Restart MATLAB to switch.'], char(pe.Executable), PYTHON_PATH);
end
fprintf('CurrentPython: %s\n', char(pe.Executable));

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
NUM_GAUSSIANS = 100;
BASE_SEED     = 42;
D             = 2;
DEVICE        = 'cuda';
NUM_TRIALS    = 11;
ETA_VALUES    = [0.2, 0.4, 0.6, 0.8, 1.0];
NUM_SAMPLES   = 500 * D;

CKPT_PATH = '.\res_rq2\msg_ela\results_max_max_min_2d.pt';
CSV_FILE  = 'IGDX_scores_max_max_min_2d.csv';

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

    % Epsilon-local optima in decision space (eps = 0)
    [pop_filtered, ~] = get_epsilon_local_optima( ...
        X_local, F_local, F_global, 0.0);
    n_eps_local = size(pop_filtered, 1);

    % --- Run HREA across seeds and eta values ---
    for trial = 1:NUM_TRIALS
        seed = BASE_SEED + trial;
        if isKey(completed, make_key(idx, seed))
            continue;
        end

        rng(seed);
        py.random.seed(seed);

        scores = zeros(1, numel(ETA_VALUES));
        for k = 1:numel(ETA_VALUES)
            ALG     = custom_HREA('save', 0);
            ALG.p   = 1.0;
            ALG.eta = ETA_VALUES(k);
            ALG.eps = 0.1;
            ALG.Solve(PRO);

            archive_decs = vertcat(ALG.Archive.dec);
            scores(k)    = IGDX(archive_decs, pop_filtered);
        end
        fprintf('idx=%d, trial=%d/%d finished\n', idx, trial, NUM_TRIALS);

        append_row(CSV_FILE, idx, seed, n_eps_local, ...
                   num_local_optima, fdc, dispersion, scores);

        completed(make_key(idx, seed)) = true;
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

function append_row(csv_file, idx, seed, n_eps, n_local, fdc, disp10, scores)
    % Append a result row to CSV; write header on first call.
    if ~isfile(csv_file)
        header = {'idx','seed','num_epsilon_local_optima', ...
                  'num_local_optima','fdc','disp_10pct', ...
                  'score_0_2','score_0_4','score_0_6','score_0_8','score_1_0'};
        writecell(header, csv_file);
    end
    row = [{idx}, {seed}, {n_eps}, {n_local}, {fdc}, {disp10}, num2cell(scores)];
    writecell(row, csv_file, 'WriteMode', 'append');
end