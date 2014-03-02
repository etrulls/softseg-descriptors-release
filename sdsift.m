function [descs mask] = siftseg4(im,seg,settings)
% [descs mask] = siftseg4(im,seg,settings)
%
% Compute segmentation-aware SIFT.
% INPUTS
% im:       RGB/grayscale image
% seg:      Soft segmentations (image height x image width x segm. layers)
%           Leave empty ([]) to get plain SIFT features
% settings: See this file for details
%
% OUTPUTS
% descs: (Segmentation-aware) SIFT descriptors
% masks: 4x4 masks for every image pixel (optional)
%
% Copyright (C) 2012 Eduard Trulls
% etrulls@iri.upc.edu
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

%% init
if nargin~=3
	error('Bad parameters');
end

if isempty(seg)
	use_seg = false;
	if nargout==2
		error('Only one output for this mode');
	end
else
	use_seg = true;
end

%% check
assert(settings.lambda>=0);
[h w c] = size(im);
if c>1
	im = rgb2gray(im);
end

if use_seg
	assert(h == size(seg,1) && w == size(seg,2));
	num_seg = size(seg,3);
end

if round(settings.scale) ~= settings.scale || settings.scale < 1
	error('The bin size must be an integer (to do: subsample the segmentation further)');
end

%% pad image
pad_size = 2*settings.scale;%+2;
im_padded = padarray(im,[pad_size pad_size],'symmetric');
if use_seg
	seg_padded = padarray(seg,[pad_size pad_size],'symmetric');
end

frames = zeros(4,h*w);
count = 0;
[x y] = meshgrid(1:w,1:h);
frames = [x(:)'+pad_size; y(:)'+pad_size; repmat(settings.scale,1,numel(y)); repmat(0,1,numel(y)) ];

%% get descriptors
magnif = 1;
tic;
[dummy descs] = vl_sift(single(im_padded),'frames',frames,'magnif',1);
fprintf('Computed descriptors in %.02f\n',toc);

% convert from uint8
descs = double(descs);

% normalize
descs = descs ./ repmat(sqrt(sum(descs.^2,1)),[128,1]);
descs(find(isnan(descs))) = 0;

%% compute and apply masks
if use_seg
	t_masks = tic;
	SBP = magnif * settings.scale; % bin size
	NBP = 4; % number of hor/vert bins
	
	% precompute block values
	% subsample to get bin centered at half pixels
	seg_padded_ss = imresize(seg_padded,2);
	f = ones(2*SBP) / (2*SBP)^2;
	blocks = zeros(size(seg_padded_ss,1),size(seg_padded_ss,2),num_seg);
	for k=1:num_seg
		blocks(:,:,k) = conv2(seg_padded_ss(:,:,k),f,'same');
	end
	
	% get masks
	[x y] = meshgrid([-3 -1 1 3]*SBP);
	d = zeros(h*w,num_seg,16);
	r = size(blocks);
	for s=1:num_seg
		for k=1:16
			% we use the average over SBPxSBP around the pixel as a reference value
			d(:,s,k) = blocks(2*frames(2,:)+y(k) + (2*frames(1,:)+x(k)-1)*r(1) + (s-1)*r(1)*r(2)) - blocks(2*frames(2,:) + (2*frames(1,:)-1)*r(1) + (s-1)*r(1)*r(2));
		end
	end
	d = squeeze(sqrt( sum(d.^2,2) ));
	mask = exp(-settings.lambda * d);

	% re-normalize
	if settings.norm_mask
		mask = mask ./ repmat(sum(mask,2),[1 16]) * 16 ;
	end
	
	fprintf('Computed segmentation masks in %.02f\n',toc(t_masks));
	
	% apply masks
	% transpose because vlsift follows row-major order
	mask = reshape(mask,[h*w 4 4]);
	m = permute(mask,[1 3 2]);
	m = reshape(m,h*w,16)';
	m = repmat(m(:)',[8 1]);
	descs = descs(:) .* m(:);

	mask = reshape(mask,[h w 4 4]);
	
	if ~settings.norm_mask
		descs = reshape(descs,128,h*w);
		% renormalize or the masks could get very small values
		descs = descs ./ repmat(sqrt(sum(descs.^2,1)),[128,1]);
		descs(find(isnan(descs))) = 0;
	end
end

descs = permute(reshape(descs,[128,h,w]),[2 3 1]);
