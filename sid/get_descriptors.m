function [invar,polar,grd,X,Y] = get_descriptors(inim,settings,fc,X,Y)
% [polar,invar,grd,X,Y] = get_descriptors(inim,settings,fc,X,Y);
%
% Main function for scale-invariant descriptor construction
% See demo scripts for sparse- and dense- descriptor usage.
%
% Inputs
% inim          : input image
% settings      : settings struct for descriptor construction
% fc            : spacing of points, in pixels (1: fully dense descriptors)
%                 if empty, descriptor locations need to be specified.
% X,Y           : descriptor coordinates (used only if fc is empty)
%
% Outputs
% polar         : normalized orientation- and scale- covariant descriptor
% invar         : invariant descriptor
% grd           : grid used for descriptor construction
% [X,Y]         : locations where descriptors were constructed
%                 do not use if you want image-formated output
%
% Copyright (C) 2012  Iasonas Kokkinos
% iasonas.kokkinos@ecp.fr
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%


if ~isfield(settings,'dob'),
    settings.dob = 0;
end
if ~isfield(settings,'invar'),
    settings.invar  = 'both';
end

if size(inim,3)==3,
    inim = rgb2gray(inim);
end
inim = single(inim);

%% convolution results used for daisy -like descriptors
[dzy,grd,settings]  = init_dzy(inim,settings);
[szv,szh,dm]        = size(inim);

if isempty(fc)
    X         = single(min(max(round(X),1),szh)-1);
    Y         = single(min(max(round(Y),1),szv)-1);
else
    [X,Y]     = meshgrid([0:fc:szh-1],[0:fc:szv-1]);
    [sz(1),sz(2)] = size(X);
    X         = X(:)';
    Y         = Y(:)';
end

desc     = mex_compute_all_descriptors(dzy.H, dzy.params, dzy.ogrid(:,:,1), dzy.ostable, single(0),single(X'),single(Y'))';
nf       = dzy.HQ;          % number of orientations x polarities
nr       = settings.nrays;  % number of rays
ns       = length(dzy.cind);% number of scales (radii)

% matlab-lisp: make polar and normalize descriptor
polar =  make_polard(reshape(desc',[nf,nr,ns,size(desc,1)]),settings.nors);
polar =  normalize_polard(polar,settings.use_nr); 

if nargout>=2,
    if isfield(settings,'reshape'),
        if settings.reshape
            szp             = size(polar);
            polar           = reshape(polar,[szp(1:end-1),sz(1),sz(2)]);
        end
    end
    %return;
end

invar               = get_desc(polar,settings.cmp,settings.invar);
%% dense descriptors: reshape to be in original image size
if ~isempty(fc)
    szd             = size(invar);
    szp             = size(polar);
    invar           = reshape(invar,[szd(1:end-1),sz(1),sz(2)]);
    polar           = reshape(polar,[szp(1:end-1),sz(1),sz(2)]);
end

function descriptor =  make_polard(descriptor,nors,dog)
%% `rotate' directional derivatives so that orientations become relative
%% to ray's angle (Fig. 3 in paper)

[nf,nrays,nsc,np]   = size(descriptor);
if nf==2,
    %% steerable filter response; steer using analytic formula
    descriptor          = reshape(descriptor,[nf,nrays,nsc*np]);
    
    %% for polarity: separate positive and negative orientation-sensitive responses
    descriptor_out      = zeros([2*nors,nrays,nsc*np],'single');
    for r = 1:nrays,
        steering_matrix(:,1)   = cos((([0:nors-1]/nors*pi) + (r-1)/nrays*2*pi));
        steering_matrix(:,2)   = sin((([0:nors-1]/nors*pi) + (r-1)/nrays*2*pi));
        steering_matrix        = single(steering_matrix);
        %% steer derivative filters
        prd                     = steering_matrix*squeeze(descriptor(:,r,:));
        %% polarize
        descriptor_out(:, r, :) = max([prd;-prd],0);
    end
    descriptor = reshape(descriptor_out,[2*nors,nrays,nsc,np]);
else
    descriptor          = reshape(descriptor,[nf,nrays,nsc*np]);
    descriptor_out      = zeros([nf,nrays,nsc*np],'single');
    fracshifts          = nf*[0:nrays-1]/nrays;
    for r = 1:nrays,
        in = squeeze(descriptor(:,r,:));
        transform_matrix  = rotate_weights(nf,fracshifts(r));
        prm = transform_matrix*in;
        descriptor_out(:,r,:) = prm;
    end
    descriptor = reshape(descriptor_out,[nf,nrays,nsc,np]);
end

function t = get_desc(feats,cmp,invar)
%% FT of descriptors

%% normalization, so that points around the boundaries get a bit boosted
%% (otherwise their FT will be lower, due to the smaller number of non-zero
%% observations)

sz = size(feats); 
for it =size(feats,1):-1:1
    switch invar
        case 'both',
            dc = abs(fft(fft(squeeze(feats(it,:,:,:)),[],2),[],1));
            %dc = dc.*repmat(factor,[1,1,size(dc,3)]);
        case 'scale'
            dc = abs(fft(squeeze(feats(it,:,:,:)),[],2));
        case 'angle',
            dc = abs(fft(squeeze(feats(it,:,:,:)),[],1));
        case 'raw'
            dc = squeeze(feats(it,:,:,:));
    end
    if cmp==0
			if strcmp(invar,'scale')
				dc = dc(:,2:ceil([end/2]),:);
			elseif ~strcmp(invar,'raw')
				dc = dc(2:ceil([end/2]),:,:);
			end
		else
			if strcmp(invar,'scale')
				dc = dc(:,2:ceil([end/2]),:);
			elseif strcmp(invar,'angle')
				dc = dc(2:ceil([end/2]),:,:);
			elseif ~strcmp(invar,'raw')
				dc = dc(2:ceil([end/2]),setdiff([1:end],ceil(end/2)+[-6:7]),:);
			end
			%dc = dc(2:(end/2-6),setdiff([1:end],ceil(end/2)+[-6:7]),:);
    end
    t(it,:,:,:) = dc;
end


function [feats_match] = normalize_polard(feats_match,donorm)
%% normalize polar descriptors separately per ring
[nf,nrays,ns,np] = size(feats_match);
%rps  = zeros(size(feats_match));
if (donorm==1)
    for sc = 1:ns,
        fsc    = feats_match(:,:,sc,:);
        nrm_sc = (sum(sum(pow_2(fsc),1),2));
        %% ratio of observed points in ring (max with 1/nrays for robustness)
        cnt_sc = max(sum(any(fsc,1),2),1);
        %% energy in ray (normalized by # of observations)
        nrm_sc = max(sqrt(nrm_sc./cnt_sc),.1);
        feats_match(:,:,sc,:) = fsc./repmat(nrm_sc,[nf,nrays,1,1]);
    end
elseif (donorm==2)
	sz = size(feats_match);
	% single point fix
	if length(sz) == 3
		sz(4) = 1;
	end
	
	%nrm_all     = max(sqrt(sum(sum(sum(pow_2(feats_match),1),2),3)),.1);
	%nrm_all     = max(sqrt(sum(sum(sum(feats_match.*feats_match,1),2),3)),.1);
	%%{
	nrm_all = zeros(1,1,1,sz(4),'single');
	for i=1:sz(1)
		for j=1:sz(2)
			for k=1:sz(3)
				nrm_all = nrm_all + feats_match(i,j,k,:).^2;
			end
		end
	end
	nrm_all = max(sqrt(nrm_all),.1);
	%%}
	%feats_match = feats_match./repmat(nrm_all,[sz(1),sz(2),sz(3),1]);
	for i=1:sz(1)
		for j=1:sz(2)
			for k=1:sz(3)
				feats_match(i,j,k,:) = feats_match(i,j,k,:) ./ nrm_all;
			end
		end
	end
end

    

function r = un(i)
r = i - 10*floor(i/10);

function r  = pow_2(x)
r = x.*x;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Daisy code, original by Tola, adapted for this project by Iasonas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [dzy,grd,settings] = init_dzy(imin, settings)
%%
SI = 1;
LI = 1;
NT = 1;

[grd,grid,ogrid,nrays,sgs,sc_sig] = get_grid(settings);
csigma      = sgs*sc_sig;
cind        = 1:length(sgs);
cn          = length(cind);

im  = single(imin);
h   = size(im,1);
w   = size(im,2);
no  = size(im,3); 

switch  settings.dog
    case 0
        dob = 1;
        fct = 1;
        nors    = settings.nors;   %% how many orientations are estimated
        nors_ft = nors;            %% >> are stored
        consec  = (size(im,3)/nors)/2;
    case 1
        dob = 0;
        fct = 2;
        ors = pi*[0:settings.nors-1]/settings.nors;
        ors = [ors,ors+pi];
        nors    = 2;
        nors_ft = 2;
        settings.nors_ft    = nors_ft;
    case {2}
        nors_ft  = 8;
        cu0      = csigma(1);
        CL_now   = (layered_gradient( im, nors_ft,cu0));
end


dim = double(im); 
%% exploit x-y separability of directional derivative operator

dzy.H   = zeros(h*w,nors_ft,cn,'single');
for r=1:cn
    cu  = max(csigma(r),.5);
    switch settings.dog,
        case {2}
            if r==1
                sig_inc = sqrt(csigma(r)^2 - cu0^2);
                factor  = sqrt(csigma(r));
                CL_now = CL_now;
            else
                sig_inc = sqrt(csigma(r)^2 - csigma(r-1)^2 );
                factor  = sqrt(csigma(r)/csigma(r-1));
                CL_now  = smooth_layers(CL_now,sig_inc);
            end
            if settings.dog>=3,
                factor = 1;
            end
            CL_now = CL_now*factor;
          
            for k=1:nors_ft
                %dzy.H(:,k,r) = factor*reshape(squeeze(single(anigauss(CL_0(:,:,k),sig_inc,sig_inc,0)))',h*w,1);
                dzy.H(:,k,r) = reshape(CL_now(:,:,k)',h*w,1);
            end
        case 1
            dx  = single((cu))*single(anigauss(dim,cu,cu, -90 , 0, 1));
            dy  = single((cu))*single(anigauss(dim,cu,cu,   0 , 0, 1));
            
            dzy.H(:,1,r) = reshape(dx',h*w,1);
            dzy.H(:,2,r) = reshape(dy',h*w,1);
        case 0
            for k=1:nors,
                if 0,
                    cen = consec*2*(k-1) + 1;
                    no  =size(dim,3);
                    left = mod(cen-2,no) + 1;
                    right = mod(cen,no) + 1;
                    %in = (dim(:,:,cen) + (dim(:,:,left) + dim(:,:,right))/2)/2;
                    %slices = dim(:,:,mod(consec*2*(k-1) + [-consec:consec]-1,end)+1);
                    %in = slices(:,:,end+1/2) + .5*
                else
                    cen = consec*2*(k-1) + 1;
                    in = dim(:,:,cen);
                end
                ft  = single(anigauss(squeeze(in),cu,cu, 0 , 0, 0))*sqrt(cu)*.6;
                dzy.H(:,k,r) = reshape(ft',h*w,1);
            end
    end
end

HQ = size(dzy.H,2);
TQ = nrays;

dzy.h       = h;
dzy.w       = w;
dzy.TQ      = TQ;
dzy.HQ      = HQ;
dzy.HN      = size(grid,1);
dzy.DS      = dzy.HN*HQ;
dzy.cind    = cind;
dzy.csigma  = csigma;
dzy.ostable = compute_orientation_shift(HQ,1);
%fprintf(1,'-------------------------------------------------------\n');
dzy.SI = SI;
dzy.LI = LI;
dzy.NT = NT;
dzy.params = single([dzy.DS dzy.HN dzy.h dzy.w 0 0 TQ HQ SI LI NT length(dzy.ostable)]);
dzy.params(11) = 0;

%% skip the first element
dzy.ogrid           = ogrid(2:end,:,:);
dzy.grid            = grid(2:end,:,:);
dzy.HN              = dzy.HN - 1;
dzy.params(2)       = dzy.params(2) - 1;
dzy.params(1)       = dzy.params(1) - 1*dzy.params(end-4);


% computes the required shift for each orientation
function ostable=compute_orientation_shift(hq,res)
if nargin==1
    res=1;
end
ostable = single(0:res:359)*hq/360;


%% compute histograms

function transform_matrix = interpolate_weights(angle_ray,angs,nf)
angles_sam = angle_ray + angs;

transform_matrix = zeros(nf);
for ang_ind  = [1:length(angles_sam)]
    ang = angles_sam(ang_ind);
    ds  = abs(sin(pi*(angs - ang)/nf));
    [sr,id] = sort(ds);
    if sr(1)<1e-10,
        transform_matrix(ang_ind,id(1)) = 1;
    else
        sd = 1./sr([1,2])./(sum(1./sr([1,2])));
        transform_matrix(ang_ind,id([1,2])) = sd;
    end
end


function [matr] = rotate_weights(n,a)
ns = [-300:300];
nsi = mod(ns,n)+1;
numerators   = sin(pi*(ns-a));
denominators = pi*(ns-a);
isnz = abs(denominators)>1e-5;
weights = ones(1,length(ns));
weights(isnz) = numerators(isnz)./denominators(isnz);
for k =1:(n),
    weight(k) = sum(weights.*(nsi==k));
end
idxs = [1:n];
for k=1:n,
    matr(k,:) = weight(mod(idxs - k, n)+1);
end
matr = single(matr);



function [L] = layered_gradient( im, layer_no,cu )

%% first smooth the image
[hf,df] = gaussian_1d(cu, 0, 5);
im=conv2(im, hf, 'same');
im=conv2(im, hf', 'same');
%% compute x and y derivates
hf = [1 0 -1]/2;
vf = hf';
dx = conv2(im,hf,'same');
dy = conv2(im,vf,'same');
%% compute layers
[h,w]=size(im);
L(h,w,layer_no)=single(0);
for l=0:layer_no-1
    th  = -pi/2+ 2*l*pi/layer_no;
    kos = cos(th);
    zin = sin(th);
    L(:,:,l+1) = max(kos*dx+zin*dy,0);
end
%
function [flt,dflt] = gaussian_1d(sigma, mean, fsz)
sz = floor((fsz-1)/2);
v = sigma*sigma*2;
x=-sz:sz;
flt = exp( -((x-mean).^2)/v );
n=sum(flt);
flt=flt/n;
dflt  = flt.*(-2*x/v);

%
% smooth all the layers with sigma
%

function [SL] = smooth_layers( L, sigma )

fsz = filter_size(sigma);
hf  = gaussian_1d(sigma, 0, fsz);
%
ln = size(L,3);
for l=1:ln
    SL(:,:,l) = conv2(L(:,:,l),hf,'same');
    SL(:,:,l) = conv2(SL(:,:,l),hf','same');
end

%
function fsz = filter_size( sigma )

fsz = floor(5*sigma);
if mod(fsz,2) == 0 
    fsz = fsz +1;
end
if fsz < 3
    fsz = 3;
end



