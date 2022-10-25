function aurora(cfg,xg)
arguments
    cfg (1,1) struct {mustBeNonempty}
    xg (1,1) struct {mustBeNonempty}
end

%% unpack configuration data
ymd = cfg.ymd;
UTsec0 = cfg.UTsec0;
tdur = cfg.tdur;
dtprec = cfg.dtprec;
dtE0 = cfg.dtE0;
itprec = 0:dtprec:tdur;
itE0 = 0:dtE0:tdur;
ltE0 = length(itE0);

%% unpack grid data
x2 = xg.x2(3:end-2);
x3 = xg.x3(3:end-2);
mlon = squeeze(xg.phi(end,:,1))*180/pi;
mlat = 90-squeeze(xg.theta(end,1,:))*180/pi;
llon = length(mlon);
llat = length(mlat);

%% create east-north-time grid
[X2prec,X3prec,ITprec] = ndgrid(x2,x3,itprec);
[X2E0,X3E0,ITE0] = ndgrid(x2,x3,itE0);

%% generate input data maps
[Qit,E0it,~,~,~] = aurora_map(X2prec,X3prec,ITprec,cfg);
[~,~,mapJ,mapU,mapV] = aurora_map(X2E0,X3E0,ITE0,cfg);

%% write precip files
pg.mlon = mlon;
pg.mlat = mlat;
pg.llon = llon;
pg.llat = llat;
pg.Qit = Qit;
pg.E0it = E0it;
pg.times = datetime(ymd) + seconds(UTsec0+itprec);
gemini3d.write.precip(pg,cfg.prec_dir)

%% write field files
rtime = 0;
if cfg.flagdirich % flow driven
    parpool(cfg.ap_np);
    t = UTsec0;
    disp(pad([' time = ',num2str(t),' s '],80,'both','-'))
    v2 = mapU(:,:,1);
    v3 = mapV(:,:,1);
    c2 = cfg.ap_cad2;
    c3 = cfg.ap_cad3;
    dir = cfg.E0_dir;
    Vmaxx1it = zeros(llon,llat,ltE0);
    Vmaxx1it(:,:,1) = gemscr.postprocess.flow2pot(v2,v3,c2,c3,xg,dir,numf=64,saveplt=1,time=t,usepar=1,multirun=1);
    for it = 2:ltE0 % skip first time step
        t = UTsec0 + (it-1)*dtE0;
        disp(pad([' time = ',num2str(t),' s '],80,'both','-'))
        v2 = mapU(:,:,it);
        v3 = mapV(:,:,it);
        tic
        if all(isequal(v2,mapU(:,:,it-1)),isequal(v3,mapV(:,:,it-1)))
            Vmaxx1it(:,:,it) = Vmaxx1it(:,:,it-1);
        else
            Vmaxx1it(:,:,it) = gemscr.postprocess.flow2pot(v2,v3,c2,c3,xg,dir,numf=64,saveplt=1,time=t,usepar=1,multirun=1);
        end
        rtime = (toc + (it-2)*rtime)/(it-1);
        disp(['Estimated time remaining: ',num2str((ltE0-it)*rtime/60),' min'])
    end
    delete(gcp('nocreate'))
else % current driven
    Vmaxx1it = mapJ;
end

Vmaxx1it(:,:,1) = zeros(llon,llat); % no forcing in first timestep

E.flagdirich = ones(1,ltE0)*cfg.flagdirich;
E.Exit = ones(llon,llat,ltE0)*cfg.ap_ExBg;
E.Eyit = ones(llon,llat,ltE0)*cfg.ap_EyBg;
E.Vminx1it = zeros(llon,llat,ltE0);
E.Vmaxx1it = Vmaxx1it;
E.Vminx2ist = zeros(llat,ltE0);
E.Vmaxx2ist = zeros(llat,ltE0);
E.Vminx3ist = zeros(llon,ltE0);
E.Vmaxx3ist = zeros(llon,ltE0);
E.mlon = mlon;
E.mlat = mlat;
E.llon = llon;
E.llat = llat;
E.times = datetime(ymd) + seconds(UTsec0+itE0);
gemini3d.write.Efield(E,cfg.E0_dir);

%% functions
% map function
    function [Qit,E0it,J,U,V] = aurora_map(x2,x3,it,pars)
        p = pars;
        x2 = x2 - p.driftE*it;
        x3 = x3 - p.driftN*it;
        c = (p.ctr_spn/2)*tanh(2*p.ctr_slp*(x2-p.ctr_pos)/p.ctr_spn);
        dcdx = p.ctr_slp*sech(2*p.ctr_slp*(x2-p.ctr_pos)/p.ctr_spn).^2;
        %     s = sqrt(1+dcdx.^2);
        s = 1;
        b = bar(x2,p.bar_pos+p.bar_vel*it,p.bar_frc,p.bar_gsl); % loading bar
        d = (2-p.dim_frc*(1-tanh(2*(it-p.dim_del)/p.dim_tim)))/2; % dimming
        J_amp = p.K_amp/p.J_wth;
        Qit = (p.Q_amp_h-p.Q_amp_l)*d.*b.*...
            sheet(x3,c+p.Q_wth_l/2+p.Q_off_l+p.Q_off_h*p.Q_wth_l/2,p.Q_wth_h*s,p.Q_gsl_h)...
            +(p.Q_amp_l-p.Q_floor)*...
            sheet(x3,c+p.Q_wth_l/2+p.Q_off_l,p.Q_wth_l*s,p.Q_gsl_l)...
            +p.Q_floor;
        E0it = (p.E_amp_h-p.E_amp_l)*d.*b.*...
            sheet(x3,c+p.Q_wth_l/2+p.Q_off_l+p.Q_off_h*p.Q_wth_l/2,p.E_wth_h*s,p.E_gsl_h)...
            +(p.E_amp_l-p.E_floor)*...
            sheet(x3,c+p.Q_wth_l/2+p.Q_off_l,p.E_wth_l*s,p.E_gsl_l)... % h and l in same pos as Q
            +p.E_floor;
        J = J_amp*(...
            sheet(x3,c+p.J_wth/2,p.J_wth*s,p.J_gsl)...
            -sheet(x3,c-p.J_wth/2,p.J_wth*s,p.J_gsl)...
            );
        F = p.F_amp*(...
            sheet(x3,c-p.J_wth,p.F_wth*s,p.F_gsl)...
            -sheet(x3,c          ,p.F_wth*s,p.F_gsl)...
            +sheet(x3,c+p.J_wth,p.F_wth*s,p.F_gsl)...
            );
        U = F.*(1/sqrt(1+dcdx.^2));
        V = F.*(dcdx./sqrt(1+dcdx.^2));
    end

% sheet definition
    function [v] = sheet(x3,pos,wdth,gsl)
        v = (tanh(2*(x3-pos+wdth/2)./(gsl*wdth))-tanh(2*(x3-pos-wdth/2)./(gsl*wdth)))/2;
    end

% bar definition
    function [v] = bar(x2,pos,frac,gsl)
        v = (2-frac*(1-tanh(2*(x2-pos)/gsl)))/2;
    end
end