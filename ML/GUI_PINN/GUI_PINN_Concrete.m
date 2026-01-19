function GUI_PINN_Concrete
% Simple GUI for PINN-based UHPC strength prediction
% Inputs : OPC, BFS, FlyAsh, Water, SP, CA, FA, Age
% Outputs: Predicted CS (MPa) + global RMSE, R^2 from trained model

    close all; clc;

    % ----- Load trained model -----
    % ต้องมี pinn_model.mat จาก PINN_Concrete_ver4.m
    global net muX stdX muy stdy rmse_final R2_final;

    S = load('pinn_model_cebfip_besttt.mat', ...
             'net','muX','stdX','muy','stdy','rmse_final','R2_final');
    net        = S.net;
    muX        = S.muX;
    stdX       = S.stdX;
    muy        = S.muy;
    stdy       = S.stdy;
    rmse_final = S.rmse_final;
    R2_final   = S.R2_final;

    % ----- Create main figure -----
    fig = figure('Name','PINN – UHPC Strength Predictor',...
                 'Position',[200 80 1050 550],...
                 'Color',[1 1 1],...
                 'NumberTitle','off');

    %% -------- Panel: Input --------
    pInput = uipanel(fig,'Title','Input Mix',...
        'FontSize',14,'Position',[0.02 0.10 0.45 0.85]);

    labels = {'OPC (kg/m^3)', ...
              'BFS (kg/m^3)', ...
              'Fly Ash (kg/m^3)', ...
              'Water (kg/m^3)', ...
              'SP (kg/m^3)', ...
              'CA (kg/m^3)', ...
              'FA (kg/m^3)', ...
              'Age (days)'};

    defaults = [445 0 0 178 2.2 1126 666 28];

    editBox = gobjects(1,8);
    for i = 1:8
        uicontrol(pInput,'Style','text','String',labels{i},...
            'HorizontalAlignment','left',...
            'FontSize',14,...
            'Position',[20 420-(i-1)*45 150 25]);
        editBox(i) = uicontrol(pInput,'Style','edit',...
            'String',num2str(defaults(i)),...
            'BackgroundColor',[1 1 1],...
            'FontSize',16,...
            'Position',[180 420-(i-1)*45 120 28]);
    end

    %% -------- Panel: Prediction / Metrics --------
    pOut = uipanel(fig,'Title','Prediction with PINN',...
        'FontSize',18,'Position',[0.50 0.55 0.48 0.40]);

    % Compressive strength
    uicontrol(pOut,'Style','text','String','Compressive Strength (MPa)',...
        'HorizontalAlignment','left','FontSize',11,...
        'Position',[20 100 220 30]);

    txt_fc = uicontrol(pOut,'Style','edit','String','---',...
        'FontSize',16,'FontWeight','bold',...
        'BackgroundColor',[0.9 1.0 0.9],...
        'Position',[250 100 120 40]);

    % % RMSE / R2 (global from training dataset)
    % uicontrol(pOut,'Style','text','String','RMSE (dataset):',...
    %     'HorizontalAlignment','left','FontSize',10,...
    %     'Position',[20 55 110 25]);
    % txt_rmse = uicontrol(pOut,'Style','text',...
    %     'String',sprintf('%.4f',rmse_final),...
    %     'FontSize',14,...
    %     'Position',[130 55 80 25]);
    % 
    % uicontrol(pOut,'Style','text','String','R^2 (dataset):',...
    %     'HorizontalAlignment','left','FontSize',10,...
    %     'Position',[230 55 110 25]);
    % txt_r2 = uicontrol(pOut,'Style','text',...
    %     'String',sprintf('%.4f',R2_final),...
    %     'FontSize',14,...
    %     'Position',[340 55 80 25]);

    %% -------- Buttons --------
    btnPredict = uicontrol(fig,'Style','pushbutton','String','Predict',...
        'FontSize',14,'Position',[520 150 140 45],...
        'Callback',@predict_callback);

    btnClear = uicontrol(fig,'Style','pushbutton','String','Clear',...
        'FontSize',14,'Position',[680 150 140 45],...
        'Callback',@clear_callback);

    btnSave = uicontrol(fig,'Style','pushbutton','String','Save',...
        'FontSize',12,'Position',[840 150 140 45],...
        'Callback',@save_callback);

    uicontrol(fig,'Style','pushbutton','String','Exit',...
        'FontSize',12,'Position',[680 90 140 40],...
        'Callback',@(src,evt) close(fig));

    % % Dummy so they are used (keeps code-analyzers quiet)
    % txt_rmse; txt_r2; btnPredict; btnClear; btnSave;

    %% ================== Callbacks =====================

    % ---- Predict button ----
    function predict_callback(~,~)
        vals = zeros(1,8);
        for k = 1:8
            vals(k) = str2double(editBox(k).String);
        end
        if any(isnan(vals))
            errordlg('Please enter numeric values for all inputs.','Input Error');
            return;
        end

        OPC   = vals(1);
        BFS   = vals(2);
        FlyA  = vals(3);
        Water = vals(4);
        SP    = vals(5);
        CA    = vals(6);
        FA    = vals(7);
        Age   = vals(8);

        % ----- Physics-based features -----
        Binder = OPC + BFS + FlyA;
        WB     = Water / Binder;
        SCM    = (BFS + FlyA) / Binder;

        % Input vector (1x10) ตามโมเดล PINN
        X = [OPC BFS FlyA Water SP CA FA Age WB SCM];

        % Normalize
        Xn  = (X - muX) ./ stdX;
        xdl = dlarray(Xn','CB');

        % Predict (normalized -> denormalized)
        y_norm = forward(net, xdl, 'Outputs','output');   % 1 x 1
        y_norm = extractdata(y_norm);
        fc_pred = y_norm * stdy + muy;

        txt_fc.String = sprintf('%.2f', fc_pred);
    end

    % ---- Clear button ----
    function clear_callback(~,~)
        for k = 1:8
            editBox(k).String = '';
        end
        txt_fc.String = '---';
    end

    % ---- Save button ----
    function save_callback(~,~)
        prompt = {'Project Name:'};
        dlgtitle = 'Save Prediction';
        answer = inputdlg(prompt, dlgtitle, [1 40], {'project1'});
        if isempty(answer), return; end
        pname = answer{1};

        % อ่านค่าปัจจุบัน
        vals = cell(1,8);
        for k = 1:8
            vals{k} = editBox(k).String;
        end

        fname = [pname '_PINN_prediction.txt'];
        fid = fopen(fname,'w');
        fprintf(fid,'Project: %s\n', pname);
        fprintf(fid,'-------------------------------\n');
        for k = 1:8
            fprintf(fid,'%s = %s\n', labels{k}, vals{k});
        end
        fprintf(fid,'Predicted CS (MPa) = %s\n', txt_fc.String);
        % fprintf(fid,'Model RMSE (dataset) = %.4f\n', rmse_final);
        % fprintf(fid,'Model R^2 (dataset)   = %.4f\n', R2_final);
        fclose(fid);

        msgbox(sprintf('Saved to %s', fname),'Saved');
    end

end
