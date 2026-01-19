function [MultiObj, params] = make_objective_concrete(target_strength, use_python, varargin)
% Build MultiObj + params for MOPSO (cost, CO2, energy) with strength constraint
% target_strength : scalar MPa
% use_python      : true -> use sklearn model via MATLAB-Python; false -> MATLAB GPR (predict_strength_matlab)
% varargin        : {model_pkl, scaler_pkl, python_exe} for Python mode

    % ---------- Material data (ordered to match decision vector) ----------
    names   = {'OPC','BFS','FlYASH','WATER','SP','CA','FA','AGE'}; %
    cost    = [0.08, 0.10, 0.60, 0.04, 1.41, 0.01, 0.02, 0.00];
    co2     = [0.84, 0.019, 0.009, 0.0003, 0.75, 0.0048, 0.0060, 0.00];
    energy  = [4.727, 1.588, 0.833, 0.006, 18.3, 0.083, 0.114, 0.00];

    % ---------- Bounds (same order as above) ----------
    lb = [102,   0.1,   0.1, 121.75,  0.1,  801,  594, 28];
    ub = [540, 359.4, 200.1, 247.00, 32.2, 1145, 992.6, 28];  % Age fixed at 28

    MultiObj.nVar    = numel(lb);
    MultiObj.var_min = lb(:).';
    MultiObj.var_max = ub(:).';

    % ---------- Predictor selection ----------
    if use_python
        model_pkl  = get_opt(varargin, 1, 'PINN_model2.pkl');
        scaler_pkl = get_opt(varargin, 2, 'scaler2.pkl');
        py_exe     = get_opt(varargin, 3, '');
        if ~isempty(py_exe), pyenv('Version', py_exe); end
        predictor = @(x) predict_strength_python(x, model_pkl, scaler_pkl);
    else
        % MATLAB GPR predictor (train once via train_gpr_matlab, then use the .mat)
        predictor = @(x) predict_strength_matlab(x, 'PINN_matlab.mat');
    end

    % ---------- Constraint handling: penalty on violation ----------
    penalty_scale = 1e4;  % tune if needed
    MultiObj.fun  = @(X) batch_objectives(X, cost, co2, energy, target_strength, predictor, penalty_scale);

    % ---------- MOPSO params (reasonable defaults) ----------
    params.Np     = 100;
    params.Nr     = 30;
    params.maxgen = 300;
    params.W      = 0.4;
    params.C1     = 2.0;
    params.C2     = 2.0;
    params.ngrid  = 20;
    params.maxvel = 10;   % % of range
    params.u_mut  = 0.5;

end  % ===== END main function =====


% ===================== Local helpers (same file) =====================

function F = batch_objectives(X, cost, co2, energy, target_strength, predictor, penalty_scale)
    if isvector(X), X = X(:).'; end
    N = size(X,1);
    f_cost   = X * cost(:);
    f_co2    = X * co2(:);
    f_energy = X * energy(:);

    pred = zeros(N,1);
    for i = 1:N
        pred(i) = predictor(X(i,:));
    end
    viol = max(0, target_strength - pred);
    pen  = penalty_scale * viol;

    F = [f_cost + pen, f_co2 + pen, f_energy + pen];
end

function out = get_opt(cellargs, idx, default)
    if numel(cellargs) >= idx && ~isempty(cellargs{idx}), out = cellargs{idx};
    else, out = default;
    end
end
