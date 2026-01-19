function REP = GMEO(params,MultiObj)
%
% Usage: REP = GMEO(params, MultiObj)
%   params.Np, params.Nr, params.maxgen, params.ngrid, params.u_mut (optional)
%   MultiObj.fun (row-wise vectorized), MultiObj.nVar, var_min, var_max
%
% Returns:
%   REP.pos      : archive decision vectors (nRep x nVar)
%   REP.pos_fit  : archive objective values (nRep x M)
%   REP.hypercube_limits, REP.grid_idx, REP.grid_subidx, REP.quality

    %% ---------- Params & problem ----------
    Np      = params.Np;
    Nr      = params.Nr;
    maxgen  = params.maxgen;
    ngrid   = params.ngrid;
    u_mut   = getfieldwithdef(params,'u_mut',0.05); 

    fun     = MultiObj.fun;
    dim     = MultiObj.nVar;
    lb      = MultiObj.var_min(:).';
    ub      = MultiObj.var_max(:).';

    %% ---------- Init population -----------
    rng('shuffle');
    POP = rand(Np,dim).*(ub-lb) + lb;                % positions
    FIT = fun(POP);                                  % (Np x M)
    bad = any(~isfinite(FIT),2);
    if any(bad)
        POP(bad,:) = rand(sum(bad),dim).*(ub-lb) + lb;
        FIT(bad,:) = fun(POP(bad,:));
    end
    M = size(FIT,2);

    % Initial repository
    DOM = checkDomination(FIT);
    REP.pos     = POP(~DOM,:);
    REP.pos_fit = FIT(~DOM,:);
    REP         = updateGrid(REP,ngrid);

    % Precompute spiral seeds
    alpha=1.34; beta=0.3;
    xspir = zeros(1,Np); yspir = zeros(1,Np);
    for i=1:Np
        theta=(1-10*i/Np)*pi;
        r=alpha*exp(beta*theta/3);
        xspir(i)=r*cos(theta); 
        yspir(i)=r*sin(theta);
    end

    %% ---------- Plotting (2D/3D) ----------
    h_fig = [];
    if M==2 || M==3
        h_fig = figure(1); clf;
        if M==2
            plot(FIT(:,1),FIT(:,2),'or'); hold on;
            plot(REP.pos_fit(:,1),REP.pos_fit(:,2),'ok'); grid on; axis square;
            xlabel f1; ylabel f2; try setTicksFromGrid(REP); catch, end
        else
            plot3(FIT(:,1),FIT(:,2),FIT(:,3),'or'); hold on;
            plot3(REP.pos_fit(:,1),REP.pos_fit(:,2),REP.pos_fit(:,3),'ok'); grid on; axis square;
            xlabel f1; ylabel f2; zlabel f3; try setTicksFromGrid(REP); catch, end
        end
        drawnow;
    end
    % disp(['Generation #0 - Repository size: ' num2str(size(REP.pos,1))]);

    %% ---------- Main loop -----------------
    s=0; z=0; t=0;   %
    for gen = 1:maxgen
        % --- non-dominated rank & crowding for scalar “score” ---
        rank  = nd_rank(FIT);                 % 1 = best front
        crowd = crowdingDistance(FIT);        % larger = sparser
        score = rank + 1./(1+crowd);          % smaller is better

        % Leader from archive (roulette over sparse cubes), fallback to best score
        if ~isempty(REP.pos)
            leader = REP.pos(selectLeader(REP),:);
        else
            [~,ix] = min(score);
            leader = POP(ix,:);
        end

        % Worst indiv by score
        [~,widx] = max(score);
        WorstX = POP(widx,:);

        % Diversity measures
        Dis = abs(mean(mean((POP - leader)./(WorstX - leader + eps))));
        C  = max(0, 1 - gen/maxgen);
        T  = (1 - (sin((pi*gen)/(2*maxgen))))^(gen/maxgen);
        die = (t<15) * (0.02*T) + (t>=15) * 0.02;
        w   = 1 - ((exp(gen/maxgen)-1)/(exp(1)-1))^2;  % for rotation ops

        NEW = POP;
        NEWFIT = FIT;

        for i=1:Np
            % basic randoms
            Fsgn = sign(0.5-rand); if Fsgn==0, Fsgn=1; end
            E = 1 + T + rand;    % mild expansion
            R1 = rand(1,dim); R2 = rand(1,dim); R3 = rand(1,dim); R4 = rand(1,dim);
            S = sin(pi*R4*C);
            idx = randperm(Np, min(4,Np));
            P1 = POP(idx(1),:);
            P2 = POP(idx(min(2,numel(idx))),:);
            P3 = POP(idx(min(3,numel(idx))),:);
            %P4 = POP(idx(min(4,numel(idx))),:);

            % J(i) like CCO
            J = abs(mean((POP(i,:) - leader)./(WorstX - leader + eps)));

            % ----- Operator mix  -----
            op = rand;
            cand = POP(i,:);

            if op < 0.33
                % Rotation / spiral move around a base
                Rt1 = rand(1,dim)*pi; Rt2 = rand(1,dim)*pi;
                base = (rand<0.5)*leader + (rand>=0.5)*mean([P1;P2;P3],1);
                cand = base + 2*w*Fsgn*(cos(Rt1)).*(sin(Rt2)).*(base - POP(i,:));
            elseif op < 0.66
                % Differential-like catfish move
                step = POP(i,:) - E*leader;
                DE   = C*Fsgn;
                cand = 0.5*(leader + P1) + DE*( 2*R1.*step - R2/2.*(DE*R3 - 1) );
            else
                % Leader attraction with sinus + spiral offsets
                if J > Dis
                    cand = leader + Fsgn*S.*(leader - POP(i,:));
                else
                    if Dis*abs(randn) < J
                        Cy = 1/(pi*(1+C^2));
                        cand = leader*(1 + T^5*Cy*E) + Fsgn*(S.*(leader - POP(i,:)));
                    else
                        cand = leader*(1 + T^5*normrnd(0,C^2)) + Fsgn*(S.*(leader - POP(i,:)));
                    end
                end
                % Inject spiral coordinates sometimes
                if mod(i,2)==1
                    cand = cand + T^2*yspir(i)*(1-R1).*abs(cand - POP(i,:));
                else
                    cand = cand + T^2*xspir(i)*(1-R1).*abs(cand - POP(i,:));
                end
            end

            % Occasional recombination burst
            s = s + 1;
            if s > 10
                r1 = rand;
                jx = randperm(Np, min(2,Np));
                lesp1 = r1*POP(jx(1),:) + (1-r1)*POP(jx(end),:);
                cand  = round(lesp1) + Fsgn*r1*R1/(gen^4+eps).*cand;
                s = 0;
            end

            % Death / reinit
            if rand < die
                if rand < C
                    cand = rand(1,dim).*(ub-lb) + lb;
                else
                    r1 = rand; r2 = rand;
                    bestPert = leader*(levy(1,1)*(r1>r2) + abs(randn)*(r1<=r2));
                    Upc = max(bestPert); Lowc = min(bestPert);
                    cand = rand(1,dim)*(Upc - Lowc) + Lowc;
                end
            end

            % Bounds
            cand = SpaceBound(cand, ub, lb);

            % Evaluate candidate
            fCand = fun(cand);
            if ~all(isfinite(fCand))
                % reject invalid; keep old
                continue;
            end

            % Acceptance rule (multiobjective):
            % accept if cand dominates old; if non-dominated tie, flip coin
            dom_new_over_old = all(fCand <= FIT(i,:)) && any(fCand < FIT(i,:));
            dom_old_over_new = all(FIT(i,:) <= fCand) && any(FIT(i,:) < fCand);
            if dom_new_over_old || (~dom_new_over_old && ~dom_old_over_new && rand>=0.5)
                NEW(i,:)   = cand;
                NEWFIT(i,:)= fCand;
                t = 0;
            else
                t = t + 1;
            end
        end

        % Uniform mutation on a fraction u_mut of swarm
        nmut = round(u_mut * Np);
        if nmut > 0
            idxm = randperm(Np, nmut);
            noise = 0.05*(ub - lb).*randn(nmut,dim);
            NEW(idxm,:) = SpaceBound(NEW(idxm,:) + noise, ub, lb);
            NEWFIT(idxm,:) = fun(NEW(idxm,:));
            bad = any(~isfinite(NEWFIT(idxm,:)),2);
            if any(bad)
                NEW(idxm(bad),:) = POP(idxm(bad),:);
                NEWFIT(idxm(bad),:)= FIT(idxm(bad),:);
            end
        end

        % Commit
        POP = NEW; FIT = NEWFIT;

        % Update repository
        REP = updateRepository(REP, POP, FIT, ngrid);
        if size(REP.pos,1) > Nr
            REP = deleteFromRepository(REP, size(REP.pos,1)-Nr, ngrid);
        end

        % Plot
        if ~isempty(h_fig)
            if M==2
                figure(h_fig); cla;
                plot(FIT(:,1),FIT(:,2),'or'); hold on;
                plot(REP.pos_fit(:,1),REP.pos_fit(:,2),'ok'); grid on; axis square;
                xlabel f1; ylabel f2; try setTicksFromGrid(REP); catch, end
                if isfield(MultiObj,'truePF')
                    plot(MultiObj.truePF(:,1),MultiObj.truePF(:,2),'.','color',0.8.*ones(1,3));
                end
                drawnow;
            else
                figure(h_fig); cla;
                plot3(FIT(:,1),FIT(:,2),FIT(:,3),'or'); hold on;
                plot3(REP.pos_fit(:,1),REP.pos_fit(:,2),REP.pos_fit(:,3),'ok'); grid on; axis square;
                xlabel f1; ylabel f2; zlabel f3; try setTicksFromGrid(REP); catch, end
                if isfield(MultiObj,'truePF')
                    plot3(MultiObj.truePF(:,1),MultiObj.truePF(:,2),MultiObj.truePF(:,3),'.','color',0.8.*ones(1,3));
                end
                drawnow;
            end
        end
        % disp(['Generation #' num2str(gen) ' - Repository size: ' num2str(size(REP.pos,1))]);
    end
end

%% ================= Helpers =================
function val = getfieldwithdef(s, f, d)
    if isfield(s,f), val = s.(f); else, val = d; end
end

function rank = nd_rank(F)
% Fast non-dominated ranking (rank 1 = best front)
    N = size(F,1);
    rank = zeros(N,1);
    S = cell(N,1);
    n = zeros(N,1);
    for p=1:N
        Sp = [];
        for q=1:N
            if p==q, continue; end
            if all(F(p,:)<=F(q,:)) && any(F(p,:)<F(q,:))
                Sp(end+1)=q; 
            elseif all(F(q,:)<=F(p,:)) && any(F(q,:)<F(p,:))
                n(p)=n(p)+1;
            end
        end
        S{p}=Sp;
    end
    fronts = {};
    F1 = find(n==0).';
    fronts{1}=F1; i=1;
    while ~isempty(fronts{i})
        Q = [];
        cur = fronts{i};
        for idx=1:numel(cur)
            p = cur(idx); Sp = S{p};
            for k=1:numel(Sp)
                q = Sp(k);
                n(q)=n(q)-1;
                if n(q)==0, Q(end+1)=q; end %
            end
        end
        i=i+1; fronts{i}=unique(Q);
    end
    for k=1:(numel(fronts)-1)
        rank(fronts{k})=k;
    end
    unr = rank==0; if any(unr), rank(unr)=max(rank)+1; end
end

function crowd = crowdingDistance(F)
% NSGA-II crowding distance
    K=size(F,1); M=size(F,2);
    crowd=zeros(K,1);
    if K<=2, crowd(:)=inf; return; end
    for m=1:M
        [vals,idx]=sort(F(:,m),'ascend');
        crowd(idx(1))=inf; crowd(idx(end))=inf;
        span=max(vals)-min(vals);
        if span==0, continue; end
        for k=2:K-1
            crowd(idx(k))=crowd(idx(k))+(vals(k+1)-vals(k-1))/span;
        end
    end
end

function REP = updateRepository(REP, POP, FIT, ngrid)
    DOM = checkDomination(FIT);
    REP.pos     = [REP.pos;     POP(~DOM,:)];
    REP.pos_fit = [REP.pos_fit; FIT(~DOM,:)];
    DOM = checkDomination(REP.pos_fit);
    REP.pos     = REP.pos(~DOM,:);
    REP.pos_fit = REP.pos_fit(~DOM,:);
    % deduplicate (tolerant)
    [~,u] = unique(round([REP.pos_fit, REP.pos]*1e10),'rows');
    REP.pos     = REP.pos(u,:);
    REP.pos_fit = REP.pos_fit(u,:);
    REP = updateGrid(REP, ngrid);
end

function dom_vector = checkDomination(F)
% Returns logical vector: true if row i is dominated by any other row (minimization)
    N = size(F,1);
    dom_vector = false(N,1);
    if N <= 1
        return;
    end

    % All unordered pairs (i<j)
    pairs = nchoosek(1:N,2);  % size Kx2

    % i dominates j ?
    i_dom_j = all(F(pairs(:,1),:) <= F(pairs(:,2),:), 2) & any(F(pairs(:,1),:) < F(pairs(:,2),:), 2);
    % j dominates i ?
    j_dom_i = all(F(pairs(:,2),:) <= F(pairs(:,1),:), 2) & any(F(pairs(:,2),:) < F(pairs(:,1),:), 2);

    % Mark dominated indices
    if any(i_dom_j)
        dom_vector( unique(pairs(i_dom_j, 2)) ) = true;  % j dominated by i
    end
    if any(j_dom_i)
        dom_vector( unique(pairs(j_dom_i, 1)) ) = true;  % i dominated by j
    end
end


function REP = updateGrid(REP,ngrid)
    ndim = size(REP.pos_fit,2);
    mins = min(REP.pos_fit,[],1);
    maxs = max(REP.pos_fit,[],1);
    span = maxs - mins;
    z = span==0;
    if any(z)
        epsw = max(1,abs(mins(z)))*1e-6;
        mins(z)=mins(z)-epsw; maxs(z)=maxs(z)+epsw;
        span = maxs - mins;
    end
    REP.hypercube_limits = zeros(ngrid+1,ndim);
    for d=1:ndim
        REP.hypercube_limits(:,d)=linspace(mins(d),maxs(d),ngrid+1).';
    end
    npar = size(REP.pos_fit,1);
    REP.grid_subidx = zeros(npar,ndim);
    for d=1:ndim
        REP.grid_subidx(:,d) = min(ngrid, max(1, floor( (REP.pos_fit(:,d)-mins(d))./span(d) * ngrid ) + 1 ));
    end
    if ndim==2
        REP.grid_idx = sub2ind([ngrid ngrid], REP.grid_subidx(:,1), REP.grid_subidx(:,2));
    elseif ndim==3
        REP.grid_idx = sub2ind([ngrid ngrid ngrid], REP.grid_subidx(:,1), REP.grid_subidx(:,2), REP.grid_subidx(:,3));
    else
        dims = ngrid*ones(1,ndim);
        REP.grid_idx = zeros(npar,1);
        for i=1:npar, REP.grid_idx(i)=sub2ind(dims, REP.grid_subidx(i,:)); end
    end
    % quality per occupied cube
    ids = unique(REP.grid_idx);
    occ = zeros(numel(ids),1);
    for k=1:numel(ids), occ(k)=sum(REP.grid_idx==ids(k)); end
    REP.quality = [ids, 10./occ];
end

function selected = selectLeader(REP)
    if isempty(REP.quality)
        selected = randi(size(REP.pos,1)); return;
    end
    w = REP.quality(:,2); cw = cumsum(w);
    r = rand()*cw(end);
    sel_hyp = REP.quality(find(r<=cw,1,'first'),1);
    cand = find(REP.grid_idx==sel_hyp);
    if isempty(cand), selected = randi(size(REP.pos,1));
    else, selected = cand(randi(numel(cand)));
    end
end

function REP = deleteFromRepository(REP,n_extra,ngrid)
    M=size(REP.pos_fit,2); K=size(REP.pos_fit,1);
    crowd=zeros(K,1);
    for m=1:M
        [vals,idx]=sort(REP.pos_fit(:,m),'ascend');
        span=max(vals)-min(vals);
        if span==0
            dist=zeros(K,1);
        else
            up=[vals(2:end); inf]; down=[inf; vals(1:end-1)];
            dist=(up-down)/span;
        end
        inv=zeros(K,1); inv(idx)=1:K;
        crowd=crowd+dist(inv);
    end
    crowd(~isfinite(crowd))=inf;
    [~,ord]=sort(crowd,'ascend');
    del=ord(1:n_extra);
    REP.pos(del,:)=[]; REP.pos_fit(del,:)=[]; 
    REP = updateGrid(REP,ngrid);
end

function X = SpaceBound(X, ub, lb)
    if rand < rand
        S = (X>ub) | (X<lb);
        X = (rand(size(X)).*(ub-lb)+lb).*S + X.*(~S);
    else
        X = min(max(X,lb),ub);
    end
end

function L = levy(n,m)
    Beta = 1.5;
    % Mantegna’s algorithm
    sigma_u = ( gamma(1+Beta)*sin(pi*Beta/2) / ( gamma((1+Beta)/2)*Beta*2^((Beta-1)/2) ) )^(1/Beta);
    u = sigma_u .* randn(n,m);
    v = randn(n,m);
    L = 0.05 * u ./ (abs(v).^(1/Beta));
end

function setTicksFromGrid(REP)
    L = REP.hypercube_limits; d = size(L,2);
    if d==2
        set(gca,'xtick',L(:,1)','ytick',L(:,2)');
        axis([min(L(:,1)) max(L(:,1)) min(L(:,2)) max(L(:,2))]);
    elseif d==3
        set(gca,'xtick',L(:,1)','ytick',L(:,2)','ztick',L(:,3)');
        axis([min(L(:,1)) max(L(:,1)) min(L(:,2)) max(L(:,2)) min(L(:,3)) max(L(:,3))]);
    end
end
