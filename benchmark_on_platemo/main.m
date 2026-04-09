% ============================================================
%  Setup Python Environment
% ============================================================

pythonPath = '.\.venv\Scripts\python.exe';
pe = pyenv;

if pe.Status == "NotLoaded"
    pe = pyenv('Version', pythonPath, 'ExecutionMode', 'InProcess');
else
    if ~strcmpi(pe.Executable, pythonPath)
        error(['Python environment mismatch:\n', ...
               'Current   : ', char(pe.Executable), '\n', ...
               'Requested : ', pythonPath, '\n', ...
               'Restart MATLAB to switch.']);
    end
end

disp(['CurrentPython: ', char(pe.Executable)]);

% ============================================================
%  Setup Python Module Path
% ============================================================

scriptPath = '.\x_msg';

if count(py.sys.path, scriptPath) == 0
    insert(py.sys.path, int32(0), scriptPath);
end


% ============================================================
%  Python Module Imports
% ============================================================

construct_msg_landscape_py = py.importlib.import_module('construct_msg_landscape');
py.importlib.reload(construct_msg_landscape_py);

sampling_py = py.importlib.import_module('sampling');
py.importlib.reload(sampling_py);

make_multi_objective_msg_py = py.importlib.import_module('make_multi_objective_msg');
py.importlib.reload(make_multi_objective_msg_py);

extract_features_py = py.importlib.import_module('extract_features');
py.importlib.reload(extract_features_py);

torch_py = py.importlib.import_module('torch');


% ============================================================
%  Experiment Configuration
% ============================================================

NUM_GAUSSIANS = 100;
BASE_SEED     = 42;
D             = 2;
DEVICE_STR    = 'cuda';

CKPT_PATH = '.\res_rq2\msg_ela\results_max_max_max_2d.pt';
csv_file = 'IGDX_scores_max_max_max_2d.csv';

% ============================================================
%  MSG Construction
% ============================================================

means = sampling_py.sobol_sampling( ...
    py.int(D), ...
    py.int(NUM_GAUSSIANS), ...
    pyargs('device', py.str(DEVICE_STR), 'seed', py.int(BASE_SEED)) ...
);
means = means.to(py.str(DEVICE_STR));

data = torch_py.load(CKPT_PATH);
theta_history = double(data{'theta_history'}.detach().cpu().numpy());

for idx = 1:size(theta_history, 1)

    theta = theta_history(idx, :);
    theta = py.torch.tensor(theta, pyargs( ...
        'dtype', py.torch.float32, ...
        'device', py.str(DEVICE_STR) ...
    ));

    msg = construct_msg_landscape_py.MSGLandscape(means, theta);

    num_samples = 500*D;
    theta_tensor = py.torch.tensor(theta);
    means_tensor = py.torch.tensor(means);

    ps = py.list({0.1});
    device = 'cuda';

    features = extract_features_py.compute_features(...
        py.int(num_samples), ...
        theta_tensor, ...
        means_tensor, ...
        ps, ...
        device, ...
        py.list({'optima_feature','fdc_feature','disp_feature','r2_feature'}) ...
    );

    num_local_optima = double(features{'num_local_optima'}.cpu().numpy());
    fdc = double(features{'fdc'}.cpu().numpy());
    dispersion = double(features{'disp_10pct'}.cpu().numpy());

    multi_objective_msg = make_multi_objective_msg_py.make_multi_objective_msg( ...
        pyargs( ...
            'm', py.int(2), ...
            'dim_msg', py.int(D), ...
            'function_g', msg, ...
            'pf_shape', py.str('convex') ...
        ) ...
    );

    % ============================================================
    %  MATLAB Interface for Optimization
    % ============================================================

    multi_msg = @(x) msg_wrapper(x, multi_objective_msg, DEVICE_STR);

    f1 = @(x) get_objective_function(x, multi_msg, 1);
    f2 = @(x) get_objective_function(x, multi_msg, 2);

    PRO = UserProblem('objFcn', {f1, f2}, 'D', D+1);


    % ============================================================
    %  Pareto Set Construction
    % ============================================================

    find_optima_exact = py.getattr(msg, 'find_optima_exact');
    optima = find_optima_exact(py.float(0.0));

    local_opt  = optima(1);
    global_opt = optima(3);


    local_mat  = double(local_opt{1}.detach().cpu().numpy());
    global_mat = double(global_opt{1}.detach().cpu().numpy());

    tol = 0; 

    is_included = false(size(global_mat,1),1);

    for i = 1:size(global_mat,1)
        d = vecnorm(local_mat - global_mat(i,:), 2, 2);
        is_included(i) = any(d == tol);
    end


    X_local  = sample_pareto_set(local_opt, 2);
    X_global = sample_pareto_set(global_opt, 2);

    is_included = false(size(X_global,1),1);

    for j = 1:size(X_global,1)
        d = vecnorm(X_local - X_global(j,:), 2, 2); 
        is_included(j) = any(d == tol);
    end

    F_local = multi_msg(X_local);
    F_global = multi_msg(X_global);


    is_included = false(size(F_global,1),1);

    for j = 1:size(F_global,1)
        d = vecnorm(F_local - F_global(j,:), 2, 2); 
        is_included(j) = any(d == tol);
    end


    [pop_filtered, F_pop_filtered] = get_epsilon_local_optima( ...
        X_local, F_local, F_global, 0.0 ...
    );
    LENGTH_POP_FILTERED = size(pop_filtered, 1);

    
    for i = 1:11

        seed = BASE_SEED + i;
        rng(seed);
        py.random.seed(seed);
    

        % ============================================================
        %  Optimization Execution (p=1.0, eta=0.2, eps=0.1)
        % ============================================================
        disp('Runing HREA with p=1.0, eta=0.2, eps=0.1');
        ALG = custom_HREA('save', 0);
        ALG.p   = 1.0;
        ALG.eta = 0.2;
        ALG.eps = 0.1;

        ALG.Solve(PRO);


        % ============================================================
        %  Evaluation (IGDX)
        % ============================================================

        ArchivePopulation = ALG.Archive;
        ArchiveDecs       = vertcat(ArchivePopulation.dec);

        score_0_2 = IGDX(ArchiveDecs, pop_filtered);

        disp(['IGDX = ', num2str(score_0_2)]);

        
        % ============================================================
        %  Optimization Execution (p=1.0, eta=0.4, eps=0.1)
        % ============================================================

        disp('Runing HREA with p=1.0, eta=0.4, eps=0.1');

        ALG = custom_HREA('save', 0);
        ALG.p   = 1.0;
        ALG.eta = 0.4;
        ALG.eps = 0.1;

        ALG.Solve(PRO);


        % ============================================================
        %  Evaluation (IGDX)
        % ============================================================

        ArchivePopulation = ALG.Archive;
        ArchiveDecs       = vertcat(ArchivePopulation.dec);

        score_0_4 = IGDX(ArchiveDecs, pop_filtered);

        disp(['IGDX = ', num2str(score_0_4)]);

        
        % ============================================================
        %  Optimization Execution (p=1.0, eta=0.6, eps=0.1)
        % ============================================================

        disp('Runing HREA with p=1.0, eta=0.6, eps=0.1');

        ALG = custom_HREA('save', 0);
        ALG.p   = 1.0;
        ALG.eta = 0.6;
        ALG.eps = 0.1;

        ALG.Solve(PRO);


        % ============================================================
        %  Evaluation (IGDX)
        % ============================================================

        ArchivePopulation = ALG.Archive;
        ArchiveDecs       = vertcat(ArchivePopulation.dec);

        score_0_6 = IGDX(ArchiveDecs, pop_filtered);

        disp(['IGDX = ', num2str(score_0_6)]);

        
        % ============================================================
        %  Optimization Execution (p=1.0, eta=0.8, eps=0.1)
        % ============================================================

        disp('Runing HREA with p=1.0, eta=0.8, eps=0.1');

        ALG = custom_HREA('save', 0);
        ALG.p   = 1.0;
        ALG.eta = 0.8;
        ALG.eps = 0.1;

        ALG.Solve(PRO);


        % ============================================================
        %  Evaluation (IGDX)
        % ============================================================

        ArchivePopulation = ALG.Archive;
        ArchiveDecs       = vertcat(ArchivePopulation.dec);

        score_0_8 = IGDX(ArchiveDecs, pop_filtered);

        disp(['IGDX = ', num2str(score_0_8)]);

        
        % ============================================================
        %  Optimization Execution (p=1.0, eta=1.0, eps=0.1)
        % ============================================================

        disp('Runing HREA with p=1.0, eta=1.0, eps=0.1');

        ALG = custom_HREA('save', 0);
        ALG.p   = 1.0;
        ALG.eta = 1.0;
        ALG.eps = 0.1;

        ALG.Solve(PRO);


        % ============================================================
        %  Evaluation (IGDX)
        % ============================================================
        ArchivePopulation = ALG.Archive;
        ArchiveDecs       = vertcat(ArchivePopulation.dec);

        score_1_0 = IGDX(ArchiveDecs, pop_filtered);

        disp(['IGDX = ', num2str(score_1_0)]);

        scores = [score_0_2, score_0_4, score_0_6, score_0_8, score_1_0];
        disp('IGDX scores for different eta values:');
        disp(scores);

        % ============================================================
        %  Save results to CSV
        % ============================================================

        if ~isfile(csv_file)
            header = {'idx', 'seed', 'num_epsilon_local_optima', 'num_local_optima', 'fdc', 'disp_10pct', 'score_0_2', 'score_0_4', 'score_0_6', 'score_0_8', 'score_1_0'};
            writecell(header, csv_file);
        end

        row = [{idx}, {seed}, {LENGTH_POP_FILTERED}, {num_local_optima}, {fdc}, {dispersion}, num2cell(scores)];
        % Append row to CSV
        writecell(row, csv_file, 'WriteMode', 'append');

    end

end