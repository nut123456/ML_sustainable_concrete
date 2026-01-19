function [MultiObj, params] = make_objective_concrete_const(target_strength, use_python, varargin)
% Build MultiObj + params for MO (cost, CO2, energy) with strength + mix-design constraints
% target_strength : scalar MPa
% use_python      : true -> use sklearn model via MATLAB-Python; false -> MATLAB GPR (predict_strength_matlab)
% varargin        : {model_pkl, scaler_pkl, python_exe} for Python mode
%
% Decision vector X (1 x 8):
%   X = [OPC, BFS, FlYASH, WATER, SP, CA, FA, AGE]

    % ---------- Material data (ordered to match decision vector) ----------
    names   = {'OPC', 'BFS', 'FlYASH', 'WATER', 'SP',  'CA', 'FA','AGE'}; %
    cost    = [0.08,   0.10,     0.60,    0.04, 1.41,  0.01, 0.02, 0.00];
    co2     = [0.84,  0.019,    0.009,  0.0003, 0.75,0.0048, 0.0060, 0.00];
    energy  = [4.727, 1.588,    0.833,   0.006, 18.3, 0.083, 0.114, 0.00];

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
    MultiObj.fun  = @(X) batch_objectives(X, cost, co2, energy, ...
                                          target_strength, predictor, penalty_scale);

    % ---------- MO params (reasonable defaults) ----------
    params.Np     = 200;   % population size
    params.Nr     = 60;    % repository size
    params.maxgen = 500;   % max iterations
    params.W      = 0.4;   % inertia weight
    params.C1     = 2.0;   % cognitive (particle best)
    params.C2     = 2.0;   % social (global/rep best)
    params.ngrid  = 20;    % grid per dimension (archive grid)
    params.maxvel = 10;    % % of range
    params.u_mut  = 0.5;   % mutation probability

end  % ===== END main function =====


% ===================== Local helpers (same file) =====================

function F = batch_objectives(X, cost, co2, energy, target_strength, predictor, penalty_scale)
    % X: (N x 8) = [OPC, BFS, FlYASH, WATER, SP, CA, FA, AGE]

    if isvector(X), X = X(:).'; end
    N = size(X,1);

    % ---------- Base objectives (no penalty) ----------
    f_cost   = X * cost(:);
    f_co2    = X * co2(:);
    f_energy = X * energy(:);

    % ---------- Strength constraint (target_strength) ----------
    pred = zeros(N,1);
    for i = 1:N
        pred(i) = predictor(X(i,:));  % predicted fck (MPa)
    end
    viol_strength = max(0, target_strength - pred);   % ต้อง >= target_strength
    pen_strength  = penalty_scale * viol_strength;

    % ---------- Mix design constraints (ratios + volume) ----------
    OPC     = X(:,1);
    BFS     = X(:,2);
    FLYASH  = X(:,3);
    WATER   = X(:,4);
    SP      = X(:,5);
    CA      = X(:,6);
    FA      = X(:,7);
    % AGE  = X(:,8);  % not used in constraints

    % Effective cement (binder)
    cement = OPC + BFS + FLYASH;             % binder content
    aggTot = CA + FA;                        % total aggregate (coarse + fine)
    sand   = FA;

    % --- 1) Water–cement ratio: 0.35 <= W/C <= 0.55 ---
    wcr = WATER ./ cement;
    viol_wcr = max(0, 0.35 - wcr) + max(0, wcr - 0.55);

    % --- 2) Aggregate–cement ratio: 1 <= (CA+FA)/C <= 10 ---
    acr = aggTot ./ cement;
    viol_acr = max(0, 1.0 - acr) + max(0, acr - 10.0);

    % --- 3) SP–cement ratio: 0 <= SP/C <= 0.02 ---
    spcr = SP ./ cement;
    viol_spcr = max(0, 0.0 - spcr) + max(0, spcr - 0.02);

    % --- 4) Sand ratio: 0.35 <= FA / (FA + CA) <= 0.40 ---
    denom_agg = FA + CA;
    sand_ratio = sand ./ denom_agg;
    % (lb, ub)
    viol_sand = max(0, 0.35 - sand_ratio) + max(0, sand_ratio - 0.40);

    % --- 5) Volume constraint (≈ 1 m³) ---
    % Use: WATER/1000 + cement/3150 + FA/2650 + CA/2700 + SP/1200 ≈ 1
    % Based strictly on Table 1 densities (kg/m3):
    % OPC=3150, BFS=2500, FlyAsh=2500, WATER=1000,
    % SP=1150, CA=2650, FA=2500
    volume_est = (OPC    / 3150) + ...
                 (BFS    / 2900) + ...
                 (FLYASH / 2700) + ...
                 (WATER  / 1000) + ...
                 (SP     / 1200) + ...
                 (CA     / 2700) + ...
                 (FA     / 2650);
    viol_vol = abs(volume_est - 1.0);   % equality -> absolute deviation
    %Ref:Multi objective optimization of recycled aggregate concrete based on explainable machine learning
    % รวม violation (ถ่วง volume หนักขึ้นเล็กน้อย)
    total_viol_constraints = viol_wcr + viol_acr + viol_spcr + viol_sand + 1*viol_vol;

    pen_constraints = penalty_scale * total_viol_constraints;

    % ---------- รวม penalty ทั้งหมด ----------
    pen_total = pen_strength + pen_constraints;

    % ---------- Penalized objectives ----------
    F = [f_cost   + pen_total, ...
         f_co2    + pen_total, ...
         f_energy + pen_total];
end


function out = get_opt(cellargs, idx, default)
    if numel(cellargs) >= idx && ~isempty(cellargs{idx})
        out = cellargs{idx};
    else
        out = default;
    end
end
