% Description:
%   Reconstructing 2D ion flow field from distributed flow dataset.
%   Provides potential from input flow field on gemini grid. Uses
%   parameterized, longitudinally inclined, gaussian ridges as pseudo-basis
%   functions. See reconstructor_readme.pdf for more details.
%
% Example usage:
%   [recon_phi,recon_v2,recon_v3,P] = flow2pot(v2,v3,8,8,xg,'.')
%
% Arguments:
%   gemini_v2               bag of eastward flow vectors
%   gemini_v3               bag of northward flow vectors
%   cad2                    cadence over longitude
%   cad3                    cadence over latitude
%   xg                      gemini grid
%   outdir                  output directory
%   numf = 32               (option) number of basis functions used in reconstruction
%   showplt = false         (option) show plot
%   saveplt = false         (option) save plot
%   showboundary = false    (option) show boundary plot
%   usepar = false          (option) use parallel computation
%   multirun = false        (option) when doing mutliple runs open parpool outside of function
%   verbose = false         (option) verbose output
%   time = 0                (option) associated time (seconds)
%   maxiter = 400           (option) sqcurvefit option value for "MaxIterations"
%   maxfuneval = 1e3        (option) lsqcurvefit option value for "MaxFunctionEvaluations"
%
% Dependencies:
%   matlab R2022a or higher
%   optimization toolbox
%   parallel computing toolbox (optional)
%   image processing toolbox
%
% Contact:
%   jules.van.irsel.gr@dartmouth.edu

function [recon_phi,recon_v2,recon_v3,P] = flow2pot(gemini_v2,gemini_v3,cad2,cad3,xg,outdir,options)
arguments
    gemini_v2 (:,:) single {mustBeNonempty}
    gemini_v3 (:,:) single {mustBeNonempty}
    cad2 (1,1) int16 {mustBePositive}
    cad3 (1,1) int16 {mustBePositive}
    xg (1,1) struct {mustBeNonempty}
    outdir (1,:) char
    options.numf (1,1) int16 {mustBePositive} = 32
    options.showplt (1,1) logical {mustBeNonempty} = false
    options.saveplt (1,1) logical {mustBeNonempty} = false
    options.showboundary (1,1) logical {mustBeNonempty} = false
    options.usepar (1,1) logical {mustBeNonempty} = false
    options.multirun (1,1) logical {mustBeNonempty} = false
    options.verbose (1,1) logical {mustBeNonempty} = false
    options.time (1,1) int32 {mustBePositive} = 0
    options.maxiter (1,1) int16 {mustBePositive} = 400
    options.maxfuneval (1,1) int32 {mustBePositive} = 1e3
end
global boundaryc Bmag Nm

%% grid metadata
lx2 = xg.lx(2); lx3 = xg.lx(3);
Bmag = abs(mean(xg.Bmag,'all'));
[gemini_x2,gemini_x3] = ndgrid(xg.x2(3:end-2),xg.x3(3:end-2)); % E-N distance from grid

%% find arc boundary from v2 input
if options.verbose
    fprintf('Fitting arc boundary...\n')
end
cut = 8;
edges = edge(gemini_v2(cut:end-cut+1,cut:end-cut+1),'Sobel',64); % cut off model edges
boundary = zeros(1,size(edges,1));
for ix2 = 1:length(boundary)
    boundary(ix2) = gemini_x3(1,find(edges(ix2,:),1,'first')); % grab southern-most detected edges
end

boundaryf = fit(double(gemini_x2(cut:end-cut+1,1))*1e-5,double(boundary')*1e-5,'a+b*tanh(c+d*x)','Start',[0 1 0 1]);
boundaryc = coeffvalues(boundaryf).*[1e5,1e5,1,1e-5];
arc_bound_x3 = boundaryc(1)+boundaryc(2)*tanh(boundaryc(3)+boundaryc(4)*gemini_x2(:,1));

if options.showboundary
    figure(1) % plot arc boundary
    pcolor(gemini_x2,gemini_x3,gemini_v2); shading flat; hold on
    scatter(gemini_x2(cut:end-cut+1,1),boundary,'b')
    plot(gemini_x2(:,1),arc_bound_x3,'r','LineWidth',3.0); hold off
end

%% reconstruct
if options.verbose
    fprintf('Reconstruction setup...\n')
end
if isequal(size(gemini_v2),[lx2 lx3]) && isequal(size(gemini_v3),[lx2 lx3])
    bag_x2 = gemini_x2(cut:cad2:end-cut+1,cut:cad3:end-cut+1);
    bag_x3 = gemini_x3(cut:cad2:end-cut+1,cut:cad3:end-cut+1);
    bag_v2 = gemini_v2(cut:cad2:end-cut+1,cut:cad3:end-cut+1);
    bag_v3 = gemini_v3(cut:cad2:end-cut+1,cut:cad3:end-cut+1);
else
    error('Bag of vectors does not match grid size, [' + string(lx2) + ',' + string(lx3) + '].')
end

xdata = double(reshape(cat(3,bag_x2,bag_x3),[numel(bag_x2),2])); % reshape to len(bag) x 2 array
ydata = double(reshape(cat(3,bag_v2,bag_v3),[numel(bag_v2),2]));
xdata(isnan(ydata(:,1)),:) = []; % remove nans from bag
ydata(isnan(ydata(:,1)),:) = [];
if options.verbose
    fprintf('Number of fitting elements: ' + string(size(ydata,1)) + '\n')
end

%% optimization
if options.verbose
    fprintf('Reconstructing flow...\n')
end
if options.usepar && ~options.multirun
    parpool('local');
end
tic
Nm = options.numf; % number of basis functions
P_0 = [linspace(-5,5,Nm); ones(1,Nm); ones(1,Nm); ones(1,Nm)]; % initial parameter matrix: [x3pos (100 km), x3sig (100 km), x2inc (kV / 1000 km), x2amp (kV)]
opts = optimoptions('lsqcurvefit'...
    ,'Algorithm','levenberg-marquardt'...
    ,'UseParallel',options.usepar...
    ,'StepTolerance',1e-6...
    ,'MaxIterations',options.maxiter...
    ,'MaxFunctionEvaluations',2*numel(P_0)*options.maxfuneval...
    );
P = lsqcurvefit(@(P, xdata) F(P,xdata),P_0,xdata,ydata,[],[],opts); % optimized parameter matrix
recon_time = toc;
if options.usepar && ~options.multirun
    delete(gcp('nocreate'))
end
if options.verbose
    fprintf('Reconstruction time: ' + string(recon_time) + ' seconds\n')
end

%% creating output + plot arrays
if options.verbose
    fprintf('Constructing output arrays...\n')
end
plt_xdata = double(reshape(cat(3,gemini_x2,gemini_x3),[numel(gemini_x2),2]));
recon_v = F(P,xdata); % s/c locations only
recon_vt = F(P,plt_xdata);
recon_v2 = single(reshape(recon_vt(:,1),[lx2,lx3]));
recon_v3 = single(reshape(recon_vt(:,2),[lx2,lx3]));
recon_phi = single(phi(P,gemini_x2,gemini_x3));
recon_phi = recon_phi - mean(recon_phi(:));

%% Calculating error
if options.verbose
    fprintf('Determining goodness of fit...\n')
end
reg_buf = 0; % potential region buffer
minx2 = min(bag_x2(:))-reg_buf; maxx2 = max(bag_x2(:))+reg_buf;
minx3 = min(bag_x3(:))-reg_buf; maxx3 = max(bag_x3(:))+reg_buf;
reg = (gemini_x2>minx2 & gemini_x2<maxx2 & gemini_x3>minx3 & gemini_x3<maxx3); % region of interest around s/c
error_a_v2 = (recon_v2-gemini_v2).^2; % determine square differences in region of interest
error_a_v3 = (recon_v3-gemini_v3).^2;
error_p_v2 = ((recon_v2-gemini_v2)./max(gemini_v2(:))).^2; % determine square percent error in region
error_p_v3 = ((recon_v3-gemini_v3)./max(gemini_v3(:))).^2;
if options.verbose
    fprintf('Root median square difference in v2 is ' + string(sqrt(median(error_a_v2(reg),'all'))) + ' m/s\n')
    fprintf('Root median square difference in v3 is ' + string(sqrt(median(error_a_v3(reg),'all'))) + ' m/s\n')
    fprintf('Root median square percent error in v2 is ' + string(100*sqrt(median(error_p_v2(reg),'all'))) + ' %%\n')
    fprintf('Root median square percent error in v3 is ' + string(100*sqrt(median(error_p_v3(reg),'all'))) + ' %%\n')
end

folder = 'reconstructor';
if ~exist(fullfile(outdir,folder),'dir')
    mkdir(fullfile(outdir,folder));
end
fid=fopen(fullfile(outdir,folder,'reconstructor_error.txt'),'a');
fprintf(fid,[pad([' time = ',num2str(options.time),' s '],80,'both','-'),'\n']);
fprintf(fid,'Root median square difference in v2 is ' + string(sqrt(median(error_a_v2(reg),'all'))) + ' m/s\n');
fprintf(fid,'Root median square difference in v3 is ' + string(sqrt(median(error_a_v3(reg),'all'))) + ' m/s\n');
fprintf(fid,'Root median square percent error in v2 is ' + string(100*sqrt(median(error_p_v2(reg),'all'))) + ' %%\n');
fprintf(fid,'Root median square percent error in v3 is ' + string(100*sqrt(median(error_p_v3(reg),'all'))) + ' %%\n\n');


%% plotting results
% set data plotting ranges
buf = 1.05;
qnt = 0.99;
p_range = buf*[-1,1]*quantile(abs(recon_phi(:)),qnt);
v2_range = buf*[-1,1]*quantile(abs(gemini_v2(:)),qnt);
dv_range = [0 20];

% hard coded plot parameters
ftn = 'Consolas';
fts = 8;
xlab = 'distance east [m]';
ylab = 'distance north [m]';

% common properties
reset(0)
set(0,'defaultFigurePaperUnits','inches')
set(0,'defaultTiledlayoutPadding','tight')
set(0,'defaultTiledlayoutTileSpacing','tight')
setall(0,'FontName',ftn)
setall(0,'FontSize',fts)
setall(0,'Multiplier',1)
set(0,'defaultAxesFontSizeMode','manual')
set(0,'defaultSurfaceEdgeColor','flat')

if options.showplt || options.saveplt
    figure(2)
    set(gcf,'PaperPosition',[0,0,6.5,4.5])
    tiledlayout(3,2);

    nexttile
    hold on
    pcolor(gemini_x2,gemini_x3,gemini_v2)
    quiver(bag_x2,bag_x3,bag_v2,bag_v3,'r')
    title('model flow east')
    xlabel(xlab)
    ylabel(ylab)
    clb = colorbar;
    clb.Label.String = 'flow east [m/s]';
    clim(v2_range)

    nexttile
    hold on
    pcolor(gemini_x2,gemini_x3,gemini_v3)
    quiver(bag_x2,bag_x3,bag_v2,bag_v3,'r')
    title('model flow north')
    xlabel(xlab)
    ylabel(ylab)
    clb = colorbar;
    clb.Label.String = 'flow north [m/s]';
    clim(v2_range)

    nexttile
    hold on
    pcolor(gemini_x2,gemini_x3,recon_v2)
    quiver(xdata(:,1),xdata(:,2),recon_v(:,1),recon_v(:,2),'r')
    title('recon. flow east')
    xlabel(xlab)
    ylabel(ylab)
    clb = colorbar;
    clb.Label.String = 'flow east [m/s]';
    clim(v2_range)

    nexttile
    hold on
    pcolor(gemini_x2,gemini_x3,recon_v3)
    quiver(xdata(:,1),xdata(:,2),recon_v(:,1),recon_v(:,2),'r')
    title('recon. flow north')
    xlabel(xlab)
    ylabel(ylab)
    clb = colorbar;
    clb.Label.String = 'flow north [m/s]';
    clim(v2_range)

    nexttile
    hold on
    pcolor(gemini_x2,gemini_x3,recon_phi)
    quiver(bag_x2,bag_x3,bag_v2,bag_v3,'r')
    title('recon. potential')
    xlabel(xlab)
    ylabel(ylab)
    clb = colorbar;
    clb.Label.String = 'potential [V]';
    clim(p_range)

    nexttile
    hold on
    pcolor(gemini_x2,gemini_x3,100.*sqrt(error_p_v2))
    title('flow east error')
    xlabel(xlab)
    ylabel(ylab)
    clb = colorbar;
    clb.Label.String = 'rms error [%]';
    clim(dv_range)

    if options.saveplt
        saveas(gcf,fullfile(outdir,folder,['reconstructor_it=',num2str(options.time),'s','.png']));
    end
    if ~options.showplt
        close all
    end
end

%% functions
% phi basis function
    function phi = phi(P,x2,x3)
        x3pos = P(1,:)*1e5;
        x3sig = P(2,:)*1e5;
        x2inc = P(3,:)*1e-4;
        x2int = P(4,:)*1e3;
        phi = 0;
        for m = 1:Nm
            b = boundaryc(1) + boundaryc(2)*tanh(boundaryc(3)+boundaryc(4)*x2(:,1));
            phi = phi + (x2inc(m).*x2 + x2int(m)).*exp(-((x3-x3pos(m)-b)./x3sig(m)).^2);
        end
    end

% lsqcurvefit fitting function
    function v = F(P,xdata)
        Ni = size(xdata,1); % number of vectors in bag
        x3pos = P(1,:)*1e5; % latitudinal positions [m] (P entries are near unity)
        x3sig = P(2,:)*1e5; % latitudinal widths [m]
        x2inc = P(3,:)*1e-4; % longitudenal slope of potential ridge [V/m]
        x2amp = P(4,:)*1e3; % central amplitude of potential ridge [V]
        E = zeros(Ni,2);
        v = zeros(Ni,2);
        for i = 1:Ni % iterate through bag of vectors
            x2 = xdata(i,1);
            x3 = xdata(i,2);
            b = boundaryc(1) + boundaryc(2)*tanh(boundaryc(3)+boundaryc(4)*x2);
            db = boundaryc(2)*boundaryc(4)*sech(boundaryc(3)+boundaryc(4)*x2)^2;
            % caluclate the elements of -grad(phi) = -sum_m grad(phi_m) (see doccumentation for details)
            for m = 1:Nm % iterate through number of basis functions
                expf = exp(-((x3-x3pos(m)-b)/x3sig(m))^2);
                E(i,1) = E(i,1) + (-x2inc(m)-(2/x3sig(m)^2)*(x2inc(m)*x2+x2amp(m))*(x3-x3pos(m)-b)*db)*expf;
                E(i,2) = E(i,2) + (2/x3sig(m)^2)*(x2inc(m)*x2+x2amp(m))*(x3-x3pos(m)-b)*expf;
            end
            v(:,1) = -E(:,2)./Bmag;
            v(:,2) =  E(:,1)./Bmag;
        end
    end

% set default property type
    function setall(obj,property_suffix,val)
        property_suffix = char(property_suffix);
        l = length(property_suffix);
        properties = fieldnames(get(obj,'factory'));
        for n = 1:numel(properties)
            p = properties{n};
            if strcmp(p(end-l+1:end),property_suffix)
                set(obj,['default',p(8:end)],val)
            end
        end
    end
end
