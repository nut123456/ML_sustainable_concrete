function Concrete_MO_GUI_ver4
% GUI สำหรับ Multi-objective Concrete Mix Optimization
% Layout ปรับให้สวยงามสำหรับงานวิชาการ

close all; clc;

set(0,'defaulttextinterpreter','none');
set(0,'defaultAxesTickLabelInterpreter','none');
set(0,'defaultLegendInterpreter','none');

%% MAIN FIGURE
fig = figure('Name','Concrete Mix Optimization (GMEO)', ...
             'Position',[200 80 1150 550], ...
             'Color',[1 1 1], ...
             'NumberTitle','off');

%% LEFT PANEL
pLeft = uipanel(fig,'Title','Sustainable concrete design', ...
    'FontSize',14, 'Position',[0.02 0.08 0.48 0.88]);

%% Input fc'
uicontrol(pLeft,'Style','text','String','Target strength f''c (MPa):', ...
    'HorizontalAlignment','left', ...
    'FontSize',13, ...
    'Position',[25 360 200 30]);

editCS = uicontrol(pLeft,'Style','edit', ...
    'String','40', ...
    'FontSize',13, ...
    'Position',[230 365 80 30]);

btnRun = uicontrol(pLeft,'Style','pushbutton', ...
    'String','Run optimization', ...
    'FontSize',13, ...
    'Position',[330 365 140 30], ...
    'Callback',@run_callback);

%% Optimal Results
uicontrol(pLeft,'Style','text','String','Optimal results (GMEO):', ...
    'FontWeight','bold','FontSize',13, ...
    'HorizontalAlignment','left',...
    'Position',[20 315 260 25]);

mixLabels = { ...
    'OPC (kg/m³)', ...
    'BFS (kg/m³)', ...
    'Fly Ash (kg/m³)', ...
    'Water (kg/m³)', ...
    'SP (kg/m³)', ...
    'CA (kg/m³)', ...
    'FA (kg/m³)' };

mixText = gobjects(1,7);
y0 = 270;

for i = 1:7
    uicontrol(pLeft,'Style','text','String',mixLabels{i}, ...
        'HorizontalAlignment','left','FontSize',13, ...
        'Position',[35 y0-(i-1)*30 150 25]);

    mixText(i) = uicontrol(pLeft,'Style','text','String','--', ...
        'HorizontalAlignment','left','FontSize',13, ...
        'Position',[200 y0-(i-1)*30 150 25]);
end

%% Performance
uicontrol(pLeft,'Style','text','String','Performance:', ...
    'FontWeight','bold','FontSize',13, ...
    'HorizontalAlignment','left', ...
    'Position',[20 65 200 25]);

% Material Cost
uicontrol(pLeft,'Style','text','String','Material cost ($/kg):', ...
    'HorizontalAlignment','left','FontSize',12, ...
    'Position',[35 40 150 22]);

txtMC = uicontrol(pLeft,'Style','text','String','--', ...
    'HorizontalAlignment','left','FontSize',12, ...
    'Position',[200 40 120 22]);

% CO2
uicontrol(pLeft,'Style','text','String','CO₂ emissions (kg/kg):', ...
    'HorizontalAlignment','left','FontSize',12, ...
    'Position',[35 20 170 22]);

txtCO2 = uicontrol(pLeft,'Style','text','String','--', ...
    'HorizontalAlignment','left','FontSize',12, ...
    'Position',[200 20 120 22]);

% Energy
uicontrol(pLeft,'Style','text','String','Embodied (MJ/kg):', ...
    'HorizontalAlignment','left','FontSize',12, ...
    'Position',[35 0 170 22]);

txtEE = uicontrol(pLeft,'Style','text','String','--', ...
    'HorizontalAlignment','left','FontSize',12, ...
    'Position',[200 0 120 22]);

%% ปุ่ม Clear / Save / Exit
btnClear = uicontrol(pLeft,'Style','pushbutton', ...
    'String','Clear', ...
    'FontSize',12, ...
    'Position',[330 60 80 25], ...
    'Callback',@clear_callback);

btnSave = uicontrol(pLeft,'Style','pushbutton', ...
    'String','Save', ...
    'FontSize',12, ...
    'Position',[330 30 80 25], ...
    'Callback',@save_callback);

btnExit = uicontrol(pLeft,'Style','pushbutton', ...
    'String','Exit', ...
    'FontSize',12, ...
    'Position',[330 0 80 25], ...
    'Callback',@exit_callback);

%% Right-panel — Pareto axes
ax3d = axes('Parent',fig, ...
    'Position',[0.55 0.15 0.40 0.78]);
setup_axes(ax3d);

%% Allowed Algorithms
cols = [0.47 0.67 0.19];
marks = {'v'};

algos = {
    'GMEO', @GMEO_GUI
};

%% ตัวแปรเก็บผลล่าสุด (ใช้กับ Save/Clear)
lastResult = [];

%% ================== Callback functions ==================
    function run_callback(~,~)
        target_strength = str2double(editCS.String);
        if isnan(target_strength) || target_strength<=0
            errordlg('Invalid f''c input'); return;
        end

        use_python = false;
        [MultiObj, params] = make_objective_concrete_const(target_strength, use_python);

        [mixGMEO, Fgmeo] = run_once_GUI_basic(42, algos, params, MultiObj, ax3d, cols, marks);

        if isempty(mixGMEO)
            return;
        end

        % Update mix results
        for i=1:7
            mixText(i).String = sprintf('%.2f', mixGMEO(i));
        end

        % Update performance
        txtMC.String  = sprintf('%.2f',Fgmeo(1));
        txtCO2.String = sprintf('%.2f',Fgmeo(2));
        txtEE.String  = sprintf('%.f',Fgmeo(3));

        % เก็บผลล่าสุด
        lastResult.fc    = target_strength;
        lastResult.mix   = mixGMEO;
        lastResult.Fgmeo = Fgmeo;
    end

    function clear_callback(~,~)
        % เคลียร์ข้อความส่วน mix
        for i = 1:numel(mixText)
            mixText(i).String = '--';
        end
        % เคลียร์ performance
        txtMC.String  = '--';
        txtCO2.String = '--';
        txtEE.String  = '--';

        % เคลียร์กราฟ 3D และตั้งค่าใหม่
        cla(ax3d,'reset');
        setup_axes(ax3d);

        % ล้างตัวแปรผลล่าสุด
        lastResult = [];
    end

    function save_callback(~,~)
        if isempty(lastResult)
            warndlg('No optimization result to save. Please run optimization first.', ...
                    'No data');
            return;
        end

        [file,path] = uiputfile({'*.mat','MAT-files (*.mat)'}, ...
                                'Save optimization result', ...
                                'ConcreteResult.mat');
        if isequal(file,0)
            return; % ผู้ใช้กด Cancel
        end

        full = fullfile(path,file);
        result = lastResult;
        save(full,'result');

        % เซฟรูปกราฟ Pareto เป็น PNG ด้วยชื่อเดียวกัน + _pareto
        [fpath,fname] = fileparts(full);
        pngFile = fullfile(fpath,[fname '_pareto.png']);
        try
            exportgraphics(ax3d,pngFile,'Resolution',300);
        catch
            % ถ้า exportgraphics ไม่มี ให้ใช้ saveas แทน
            saveas(fig,pngFile);
        end

        msgbox('Optimization result and figure saved successfully.', ...
               'Saved');
    end

    function exit_callback(~,~)
        close(fig);
    end

end % end main function

%% ================ Helper functions =======================
function setup_axes(ax)
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'Material cost ($/kg)','FontSize',12);
    ylabel(ax,'CO₂ emissions (kg/kg)','FontSize',12);
    zlabel(ax,'Embodied (MJ/kg)','FontSize',12);
    title(ax,'Sustainable concrete trade-offs','FontSize',13);
    view(ax,[35 25]);
end

function [mixGMEO, Fgmeo] = run_once_GUI_basic(seed, algos, params, MultiObj, ax3d, cols, marks)

nAlgo = size(algos,1);
REP   = cell(nAlgo,1);
alive = false(nAlgo,1);

for k = 1:nAlgo
    funh = algos{k,2};
    try
        rng(seed);
        REP{k} = funh(params, MultiObj);
        alive(k) = isfield(REP{k},'pos_fit') && ~isempty(REP{k}.pos_fit);
    catch
        REP{k} = struct('pos',[],'pos_fit',[]);
        alive(k) = false;
    end
end

idxAlive = find(alive);
if isempty(idxAlive)
    mixGMEO = []; Fgmeo = []; return;
end

if isempty(ax3d) || ~isvalid(ax3d)
    ax3d = axes('Parent',gcf,'Position',[0.55 0.15 0.40 0.78]);
end

cla(ax3d); hold(ax3d,'on');

kneesF  = nan(nAlgo,3);
kneesX  = nan(nAlgo,8);
kneelbl = strings(nAlgo,1);

Fcells = cellfun(@(r) r.pos_fit, REP(idxAlive), 'UniformOutput', false);
Fall   = cell2mat(Fcells);
ideal  = min(Fall,[],1);
nadir  = max(Fall,[],1);

for idx = 1:numel(idxAlive)
    k  = idxAlive(idx);
    F  = REP{k}.pos_fit;
    X  = REP{k}.pos;
    nm = algos{k,1};

    c = cols(mod(k-1,size(cols,1))+1,:);
    m = marks{mod(k-1,numel(marks))+1};

    scatter3(ax3d,F(:,1),F(:,2),F(:,3),26,c,'filled','marker',m);

    [idxK,bestF] = knee_point_norm(F,ideal,nadir);
    if ~isnan(idxK)
        plot3(ax3d,bestF(1),bestF(2),bestF(3),'p','MarkerSize',14, ...
            'MarkerFaceColor',c,'MarkerEdgeColor','k');
        kneesF(k,:) = bestF;
        kneesX(k,:) = X(idxK,:);
        kneelbl(k)  = string(nm);
    end
end

%% ค้นหา knee ของ GMEO
ixG = find(kneelbl=="GMEO",1);
if isempty(ixG)
    mixGMEO=[]; Fgmeo=[]; return;
end

mixGMEO = kneesX(ixG,:);
Fgmeo   = kneesF(ixG,:);

%% วาด 3 ระนาบตั้งฉากผ่าน knee point
draw_orthogonal_planes(ax3d, Fall, Fgmeo);

end

%% knee point finder
function [idxK,bestF] = knee_point_norm(F,ideal,nadir)
if isempty(F)
    idxK=NaN; bestF=[NaN NaN NaN];
    return;
end
Fn = (F - ideal) ./ max(nadir - ideal, 1e-9);
[~,idxK] = min(vecnorm(Fn,2,2));
bestF = F(idxK,:);
end

%% วาด cutting planes ผ่าน knee point
function draw_orthogonal_planes(ax, Fall, Fknee)

if isempty(Fall) || any(isnan(Fknee))
    return;
end

% ชื่อแกน:
%   x = Material cost
%   y = CO2 emissions
%   z = Embodied energy
Cknee   = Fknee(1);
CO2knee = Fknee(2);
Eknee   = Fknee(3);

xmin = min(Fall(:,1)); xmax = max(Fall(:,1));
ymin = min(Fall(:,2)); ymax = max(Fall(:,2));
zmin = min(Fall(:,3)); zmax = max(Fall(:,3));

% ระนาบ z = Eknee (แนวนอน)
[Xz,Yz] = meshgrid(linspace(xmin,xmax,2), linspace(ymin,ymax,2));
Zz      = Eknee*ones(size(Xz));
surf(ax,Xz,Yz,Zz, ...
    'FaceAlpha',0.30, ...
    'EdgeColor','none', ...
    'FaceColor',[0.70 0.82 1.00], ...
    'HandleVisibility','off');
hold(ax,'on');

% ระนาบ x = Cknee
[Yx,Zx] = meshgrid(linspace(ymin,ymax,2), linspace(zmin,zmax,2));
Xx      = Cknee*ones(size(Yx));
surf(ax,Xx,Yx,Zx, ...
    'FaceAlpha',0.30, ...
    'EdgeColor','none', ...
    'FaceColor',[0.75 1.00 0.75], ...
    'HandleVisibility','off');

% ระนาบ y = CO2knee
[Xy,Zy] = meshgrid(linspace(xmin,xmax,2), linspace(zmin,zmax,2));
Yy      = CO2knee*ones(size(Xy));
surf(ax,Xy,Yy,Zy, ...
    'FaceAlpha',0.30, ...
    'EdgeColor','none', ...
    'FaceColor',[1.00 0.92 0.75], ...
    'HandleVisibility','off');

end
