%% ============================================================
%  PINN for Concrete Strength (CEB-FIP + Abrams) 
%  with 80/20 Train/Test Split and Metrics Reporting
%  Data file  : concrete_data_test.csv
%  Author     : (your name)

%% % เลือก extreme 10% ของ WB + extreme 10% ของ SCM
% idxWB  = WB_all < quantile(WB_all,0.10) | WB_all > quantile(WB_all,0.90);
% idxSCM = SCM_all < quantile(SCM_all,0.10) | SCM_all > quantile(SCM_all,0.90);
% 
% idxExtreme = idxWB | idxSCM;
% 
% % ใช้ 20% ของ extreme เป็น test
% ext_id = find(idxExtreme);
% N_test = round(0.20 * numel(ext_id));
% 
% idxTest = false(size(WB_all));
% idxTest(ext_id(randperm(numel(ext_id), N_test))) = true;
% 
% idxTrain = ~idxTest;
%% ============================================================
% ===== PINN Concrete Strength Prediction (80/20 split) =====
% TRAIN: MAE  = 0.370 MPa, RMSE = 1.121 MPa, R^2 = 0.9954
% TEST : MAE  = 1.494 MPa, RMSE = 2.558 MPa, R^2 = 0.9792
% 
% Saved PINN model + train/test metrics to pinn_model_cebfip.mat
% Total runtime: 252.08 s
% 
% ===== PINN Prediction for Experimental Mix =====
% Predicted CS = 52.968 MPa
% Actual CS    = 54.800 MPa
% Relative error = 3.34 %

%%
clear; clc; close all;
tTotal = tic;
%% ===================== Load concrete dataset =====================
% Expect columns:
%   OPC, BFS, FlyAsh, Water, SP, CA, FA, age, CS
data = readtable('concrete_data_test.csv');

OPC   = data.OPC;
BFS   = data.BFS;
FlyA  = data.FlyAsh;
Water = data.Water;
SP    = data.SP;
CA    = data.CA;
FA    = data.FA;
Age   = data.age;
fc    = data.CS;         % compressive strength (MPa)

%% -------- Physics-based engineered features --------
Binder = OPC + BFS + FlyA;          % total binder
WB     = Water ./ Binder;           % water–binder ratio
SCM    = (BFS + FlyA) ./ Binder;    % SCM proportion (slag+fly ash)/binder

% เก็บสำเนาไว้ใช้สำหรับ physics domain
WB_all  = WB;
Age_all = Age;
SCM_all = SCM;

% Input features for NN (N x 10)
X = [OPC BFS FlyA Water SP CA FA Age WB SCM];
y = fc;

% Remove rows with NaN or Inf
mask  = all(isfinite(X),2) & isfinite(y) & isfinite(WB_all) & isfinite(SCM_all) & isfinite(Age_all);
X     = X(mask,:);
y     = y(mask);
WB_all  = WB_all(mask);
Age_all = Age_all(mask);
SCM_all = SCM_all(mask);

%% ===================== Normalize X and y =========================
muX  = mean(X,1);              % 1 x 10
stdX = std(X,0,1);             % 1 x 10
stdX(stdX == 0) = 1;           % avoid division by zero

Xn = (X - muX) ./ stdX;        % normalized inputs

muy  = mean(y);
stdy = std(y);
if stdy == 0, stdy = 1; end

yn = (y - muy) / stdy;         % normalized strength

% dlarray ของ data ทั้งหมด (ใช้สำหรับ plot)
x_all_dl = dlarray(Xn','CB');  % 10 x N_all
y_all    = y;                  % N_all x 1

%% เลือก extreme 4.5% ของ WB + extreme 4.5% ของ SCM
idxWB  = WB_all < quantile(WB_all,0.05) | WB_all > quantile(WB_all,0.95);
idxSCM = SCM_all < quantile(SCM_all,0.05) | SCM_all > quantile(SCM_all,0.95);

idxExtreme = idxWB | idxSCM;

% ใช้ 20% ของ extreme เป็น test
ext_id = find(idxExtreme);
N_test = round(0.20 * numel(ext_id));

idxTest = false(size(WB_all));
idxTest(ext_id(randperm(numel(ext_id), N_test))) = true;

idxTrain = ~idxTest;

Xn_tr = Xn(idxTrain,:);        % train inputs (normalized)
Xn_te = Xn(idxTest,:);         % test  inputs (normalized)

y_tr  = y(idxTrain);           % train targets (MPa)
y_te  = y(idxTest);            % test  targets (MPa)

yn_tr = yn(idxTrain);          % train targets (normalized)
yn_te = yn(idxTest);           % test  targets (normalized)

% Convert TRAIN set to dlarray in 'CB' format: [features x batch_size]
x_data_dl = dlarray(Xn_tr','CB');      % 10 x N_train
y_data_dl = dlarray(yn_tr','CB');      %  1 x N_train

%% ===================== Physics Collocation Points ================
% random physics points in (WB, Age, SCM) domain (ใช้ domain ทั้ง dataset)
Nphys = 4000; %4000 good
%%
rng(1);

WB_phys  = rand(Nphys,1)  * (max(WB_all)  - min(WB_all))  + min(WB_all);
Age_phys = rand(Nphys,1)  * (max(Age_all) - min(Age_all)) + min(Age_all);
SCM_phys = rand(Nphys,1)  * (max(SCM_all) - min(SCM_all)) + min(SCM_all);

% Start from mean mix, then override Age, WB, SCM
X_phys = repmat(muX, Nphys, 1);  % Nphys x 10
X_phys(:,8)  = Age_phys;
X_phys(:,9)  = WB_phys;
X_phys(:,10) = SCM_phys;

% Normalize
X_phys_n  = (X_phys - muX) ./ stdX;
x_phys_dl = dlarray(X_phys_n','CB');     % 10 x Nphys

%% ===================== Physics constants (from CEB-FIP) ==========
% 1) f_cm (28-day mean strength) from data
idx28 = (abs(Age_all - 28) < 1e-6);
if any(idx28)
    fcm28 = mean(y(idx28));     % mean strength at 28 days
else
    % fallback: use all data mean (or Age in [27,29])
    fcm28 = mean(y);
end

% CEB-FIP parameter s (Table 5.1-9):
%   ~0.20 for rapid hardening, ~0.25 for normal hardening, ~0.38 for slow
s_cem = 0.25;     % assume normal hardening cement; user can adjust

SCM_mean   = mean(SCM_all);
alpha_scm  = 0.4;  % effect of SCM on effective f_cm (linear) (ตอนนี้ไม่ใช้จริง)

% 2) Abrams law: f_c * (WB)^b ~ constant
bAbr  = 1.5;
C_abr = mean( y .* (WB_all.^bAbr) );

%% ===================== Neural Network Architecture ===============
layers = [
    featureInputLayer(10, 'Normalization','none','Name','input')
    fullyConnectedLayer(128,'Name','fc1') %128
    tanhLayer('Name','tanh1')
    fullyConnectedLayer(64,'Name','fc2')  %64
    tanhLayer('Name','tanh2')
    fullyConnectedLayer(64,'Name','fc3')  %64
    tanhLayer('Name','tanh3')
    fullyConnectedLayer(1,'Name','output')];

lgraph = layerGraph(layers);
net = dlnetwork(lgraph);

%% ===================== Weights & Training Settings ===============

lambda_phys = 5e-2;    % ให้ data มีเสียงดังขึ้นหน่อย
lambda_abr  = 2e-3;    % Abrams ช่วย constrain WB แต่ไม่ลากแรงเกินไป
lambda_reg  = 1e-5;    % เพิ่ม reg กัน overfit บน train

numIterations = 20000;
learningRate  = 1e-2;

trailingAvg   = [];
trailingAvgSq = [];
lossHistory   = zeros(numIterations,1);
lossDataHist  = zeros(numIterations,1);
lossPhysHist  = zeros(numIterations,1);

%% ===================== Training Loop =============================
figure('Name','PINN Training Monitor','Color','w');
for iter = 1:numIterations

    [loss, gradients, lossData, lossPhys] = dlfeval(@modelLoss, ...
        net, x_data_dl, y_data_dl, x_phys_dl, ...
        muy, stdy, Age_phys, WB_phys, SCM_phys, ...
        fcm28, s_cem, SCM_mean, alpha_scm, ...
        bAbr, C_abr, lambda_phys, lambda_abr, lambda_reg);

    lossHistory(iter)  = gather(extractdata(loss));
    lossDataHist(iter) = gather(extractdata(lossData));
    lossPhysHist(iter) = gather(extractdata(lossPhys));

    % ADAM update
    [net, trailingAvg, trailingAvgSq] = adamupdate( ...
        net, gradients, trailingAvg, trailingAvgSq, iter, learningRate);

    % Print progress
    if mod(iter,500)==0
        fprintf('Iter %5d: Total=%.4e | Data=%.4e | Phys=%.4e\n', ...
            iter, lossHistory(iter), lossDataHist(iter), lossPhysHist(iter));
    end

    % Plot occasionally (ใช้ all data เพื่อดู fit รวม)
    if mod(iter,800)==0 || iter==1 || iter==numIterations
        % Predict on all data (denormalize)
        y_pred_norm_dl = forward(net, x_all_dl, 'Outputs','output');
        y_pred_norm = gather(extractdata(y_pred_norm_dl))';
        y_pred_all = y_pred_norm*stdy + muy;

        subplot(1,2,1);
        scatter(y_all, y_pred_all, 25, 'filled'); grid on; axis equal;
        hold on;
        minv = min([y_all; y_pred_all]); maxv = max([y_all; y_pred_all]);
        plot([minv maxv],[minv maxv],'k--','LineWidth',1.2);
        hold off;
        xlabel('Measured f_c (MPa)');
        ylabel('PINN-predicted f_c (MPa)');
        title(sprintf('f_c vs PINN (iter = %d)',iter));

        subplot(1,2,2);
        semilogy(1:iter, lossHistory(1:iter),'LineWidth',1.5); hold on;
        semilogy(1:iter, lossDataHist(1:iter),'--','LineWidth',1);
        semilogy(1:iter, lossPhysHist(1:iter),':','LineWidth',1);
        hold off; grid on;
        xlabel('Iteration');
        ylabel('Loss');
        legend('Total','Data','Physics','Location','best');
        title('Training Loss History');
        drawnow;
    end
end

%% ===================== Final Metrics (Train/Test) =================
% --- Predict TRAIN set ---
x_tr_dl = dlarray(Xn_tr','CB');
y_tr_pred_norm_dl = forward(net, x_tr_dl, 'Outputs','output');
y_tr_pred_norm = gather(extractdata(y_tr_pred_norm_dl))';
y_tr_pred = y_tr_pred_norm*stdy + muy;      % denormalize

% --- Predict TEST set ---
x_te_dl = dlarray(Xn_te','CB');
y_te_pred_norm_dl = forward(net, x_te_dl, 'Outputs','output');
y_te_pred_norm = gather(extractdata(y_te_pred_norm_dl))';
y_te_pred = y_te_pred_norm*stdy + muy;      % denormalize

% --- Helper metrics ---
mae_fun  = @(a,b) mean(abs(a-b));
rmse_fun = @(a,b) sqrt(mean((a-b).^2));
r2_fun   = @(a,b) 1 - sum((a-b).^2)/sum((a-mean(a)).^2);

MAE_tr  = mae_fun(y_tr, y_tr_pred);
RMSE_tr = rmse_fun(y_tr, y_tr_pred);
R2_tr   = r2_fun(y_tr, y_tr_pred);

MAE_te  = mae_fun(y_te, y_te_pred);
RMSE_te = rmse_fun(y_te, y_te_pred);
R2_te   = r2_fun(y_te, y_te_pred);

fprintf('\n===== PINN Concrete Strength Prediction (80/20 split) =====\n');
fprintf('TRAIN: MAE  = %.3f MPa, RMSE = %.3f MPa, R^2 = %.4f\n', MAE_tr, RMSE_tr, R2_tr);
fprintf('TEST : MAE  = %.3f MPa, RMSE = %.3f MPa, R^2 = %.4f\n', MAE_te, RMSE_te, R2_te);

% Save model + metrics (ใช้ TEST เป็น final)
mae_final  = MAE_te;
mse_final  = RMSE_te^2;
rmse_final = RMSE_te;
R2_final   = R2_te;

save('pinn_model_cebfip.mat', ...
     'net', 'muX', 'stdX', 'muy', 'stdy', ...
     'MAE_tr','RMSE_tr','R2_tr', ...
     'MAE_te','RMSE_te','R2_te', ...
     'mae_final', 'mse_final', 'rmse_final', 'R2_final');

fprintf('\nSaved PINN model + train/test metrics to pinn_model_cebfip.mat\n');
fprintf('Total runtime: %.2f s\n', toc(tTotal));

%% ===================== Helper Function: modelLoss ================
function [loss, gradients, lossData, lossPhys] = modelLoss( ...
    net, x_data, y_data_norm, x_phys, ...
    muy, stdy, Age_phys, WB_phys, SCM_phys, ...
    fcm28, s_cem, SCM_mean, alpha_scm, ...
    bAbr, C_abr, lambda_phys, lambda_abr, lambda_reg)

    % ----- 1) Data MSE in normalized space -----
    y_pred_norm_data = forward(net, x_data, 'Outputs','output');   % 1 x Nd
    lossData = mean((y_pred_norm_data - y_data_norm).^2, 'all');

    % ----- 2) Physics loss (CEB-FIP + Abrams) -----
    y_phys_norm = forward(net, x_phys, 'Outputs','output');        % 1 x Nphys
    y_phys = y_phys_norm*stdy + muy;                               % physical MPa

    % Prepare dlarray versions of Age, WB, SCM
    Age_row = dlarray(Age_phys');   % 1 x Nphys
    WB_row  = dlarray(WB_phys');
    SCM_row = dlarray(SCM_phys');

    % 2.1 CEB-FIP strength gain with SCM effect
    % f_cm,eff(SCM) = fcm28 * (1 + alpha_scm*(SCM - SCM_mean))
    % (ตอนนี้เซ็ต alpha_scm = 0 ภายใน fcm_eff เพื่อไม่ให้แกว่งเกินไป)
    fcm_eff = fcm28 * (1 + 0*alpha_scm*(SCM_row - SCM_mean));

    % beta_cc(t) = exp{ s [ 1 - (28/t)^0.5 ] }
    beta_cc = exp( s_cem * (1 - (28 ./ Age_row).^(0.5)) );

    f_CEB = fcm_eff .* beta_cc;

    resCEB = y_phys - f_CEB;

    % 2.2 Abrams-like relation: f_c * WB^b ≈ C_abr
    resAbr = y_phys .* (WB_row.^bAbr) - C_abr;

    lossCEB = mean(resCEB.^2, 'all');
    lossAbr = mean(resAbr.^2, 'all');

    lossPhys_raw = lossCEB + lambda_abr*lossAbr;
    lossPhys     = lossPhys_raw;    % for logging

    % ----- 3) L2 regularization on weights -----
    vals = net.Learnables.Value;
    reg = dlarray(0.0);
    for i = 1:numel(vals)
        reg = reg + sum(vals{i}.^2,'all');
    end

    % ----- 4) Total loss -----
    loss = lossData + lambda_phys*lossPhys_raw + lambda_reg*reg;

    % Backprop
    gradients = dlgradient(loss, net.Learnables);  
end
%% ===== 8. Predict experimental sample (from Sawekchai et al. [63]) =====
% Experimental mix design (as row vector)
% Columns: [OPC BFS FlyAsh Water SP CA FA age WB SCM]
OPC_exp   = 445;
BFS_exp   = 0;
FlyA_exp  = 0;
Water_exp = 178;
SP_exp    = 2.2;
CA_exp    = 1126;
FA_exp    = 666;
Age_exp   = 28;

Binder_exp = OPC_exp + BFS_exp + FlyA_exp;
WB_exp     = Water_exp / Binder_exp;
SCM_exp    = (BFS_exp + FlyA_exp) / Binder_exp;

x_exp = [OPC_exp BFS_exp FlyA_exp Water_exp SP_exp CA_exp FA_exp ...
         Age_exp WB_exp SCM_exp];

% Ground truth compressive strength
y_exp = 54.80;   % MPa

% ---- Normalize using training parameters ----
x_exp_n = (x_exp - muX) ./ stdX;

% Convert to dlarray
x_exp_dl = dlarray(x_exp_n','CB');   % 10 x 1

% ---- Predict using PINN ----
y_pred_norm_dl = forward(net, x_exp_dl, 'Outputs','output');
y_pred_norm = gather(extractdata(y_pred_norm_dl));
y_pred = y_pred_norm*stdy + muy;     % denormalize

% Relative error (%)
rel_error = abs(y_pred - y_exp) / y_exp * 100;

fprintf('\n===== PINN Prediction for Experimental Mix =====\n');
fprintf('Predicted CS = %.3f MPa\n', y_pred);
fprintf('Actual CS    = %.3f MPa\n', y_exp);
fprintf('Relative error = %.2f %%\n', rel_error);
