function [grd,grid,ogrid,nrays,sgs,sc_sig] = get_grid(settings);

sc_min = settings.sc_min;
sc_max = settings.sc_max;
nsteps = settings.nsteps;

if isfield(settings,'hbw')
    sc_sig = sig_from_h(sc_min,sc_max,nsteps,settings.hbw);
else
    sc_sig = settings.sc_sig;
end
nrays  = settings.nrays;


if isfield(settings,'viz'),
    dologp = 0;
else
    dologp = 1;
end

[grid,sgs]  = log_polar_grid(sc_min,sc_max,nsteps,nrays,dologp);
ogrid       = compute_oriented_grid(grid,360);
grd0        = squeeze(ogrid(:,:,1));
grd0        = grd0(2:end,:);
for d = [1:3],
    grd(:,:,d) = permute(reshape(grd0(:,d),[settings.nrays,length(sgs)]),[2,1]);
end

%% Auxiliary functions  (Copyright by Tola)
% rotate the grid

function ogrid = compute_oriented_grid(grid,GOR)

GN = size(grid,1);
ogrid( GN, 3, GOR )=single(0);
for i=0
    %for i = 0,
    th = -i*2.0*pi/GOR;
    kos = cos( th );
    zin = sin( th );
    for k=1:GN
        y = grid(k,2);
        x = grid(k,3);
        ogrid(k,1,i+1) = grid(k,1);
        ogrid(k,2,i+1) = -x*zin+y*kos;
        ogrid(k,3,i+1) = x*kos+y*zin;
    end
end


%% Adaptation of the grid construction routine of Tola
function [grid,Rs]= log_polar_grid(Rmn,Rmx,nr,TQ,dologp)
if dologp
    Rs = logspace(log(Rmn)/log(10),log(Rmx)/log(10),nr);
else
    Rs = linspace(3*Rmn,Rmx,nr);
end

ts = 2*pi/TQ;
RQ = length(Rs);
gs = RQ*TQ+1;

grid(gs,3)   = single(0);
grid(1,1:3)  = [1 0 0];
cnt=1;
for r=0:RQ-1
    for t=0:TQ-1
        cnt=cnt+1;
        rv = Rs(r+1);
        tv = t*ts;
        grid(cnt,1)=r+1;
        grid(cnt,2)=rv*sin(tv); % y
        grid(cnt,3)=rv*cos(tv); % x
    end
end


function f = sig_from_h(Rmn,Rmx,nr,hbw)
%ratio  = (log(Rmx./Rmn))/nr;
Rs              = logspace(log(Rmn)/log(10),log(Rmx)/log(10),nr);
alpha           = Rs(2)/Rs(1);
f               = ((alpha-1)/(alpha+1))/log(1/hbw);
