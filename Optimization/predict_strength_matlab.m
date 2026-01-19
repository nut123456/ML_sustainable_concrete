function fck = predict_strength_matlab(xrow, mdl_file)
% Predict strength using the saved MATLAB GPR model.
% xrow: 1x8 double in order [OPC BFS FlyAsh Water SP CA FA Age]

    if nargin < 2 || isempty(mdl_file), mdl_file = 'PINN_matlab.mat'; end
    persistent mdl pnames

    if isempty(mdl)
        S = load(mdl_file, 'mdl');
        mdl    = S.mdl;
        pnames = string(mdl.PredictorNames);
    end

    % Build a one-row table with the expected predictor names
    % Adjust these names if your dataset uses different ones after makeValidName
    varNames = ["OPC","BFS","FLYASH","WATER","SP","CA","FA","AGE"];
    xt_all   = array2table(xrow(:).', 'VariableNames', cellstr(varNames));

    % Reorder/select to match the model's predictors exactly
    try
        xt = xt_all(:, cellstr(pnames));
    catch
        % If names differ, show what the model expects
        error('Predictor name mismatch. Model expects: %s', strjoin(cellstr(pnames), ', '));
    end

    fck = predict(mdl, xt);
end


