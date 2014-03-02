%% dependencies
clear all
addpath sid
addpath gbd
addpath sift-flow
addpath sift-flow/mex

%% data
%resize factor
f1 = 0.5;
f2 = f1;

% SID settings
settings.sc_min = 3; % min ring radius
settings.sc_max = 201; % max ring radius
settings.nsteps = 28; % number of rings
settings.nrays = 32; % number of rays
settings.hbw = 0.45; % half-bandwidth height 
settings.nors = 4; % number of derivative orientations
settings.cmp = 0; % compress the invariant descriptor (experimental)
settings.dog = 1; % gradient (1) / gradient with polarization (2)
settings.use_nr = 2; % normalization
settings.invar = 'scale';	% SID invariance: 'both', 'angle', 'scale', 'raw' (raw is daisy-like)

% settings for the soft segmentation masks
settings.lambda = 30; % higher values result in sharper masks

% images
k = 40; % 10:10:60 (see data/car9)
im1_orig = im2double(imread('data/cars9/cars9_01.jpg'));
im2_orig = im2double(imread(sprintf('data/cars9/cars9_%d.jpg',k)));
gt1 = im2double(imread('data/cars9/cars9_01.pgm'));
gt2 = im2double(imread(sprintf('data/cars9/cars9_%d.pgm',k)));

im1 = imresize(im1_orig,f1);
im2 = imresize(im2_orig,f2);
im1 = clip(im1);
im2 = clip(im2);
[h1,w1,~] = size(im1);
[h2,w2,~] = size(im2);
gt1 = imresize(gt1,f1,'nearest');
gt2 = imresize(gt2,f2,'nearest');

%% SID vs SSID
for i=2%1:2
	if i==1
		use_seg = false;
		label = 'SID';
	else
		use_seg = true;
		label = 'SSID';
	end

	% masks
	if use_seg
		% compute the soft segmentations at the original resolution
		seg1 = softSegs(im1_orig);
		seg2 = softSegs(im2_orig);
		seg1 = clip(imresize(seg1,[h1 w1]));
		seg2 = clip(imresize(seg2,[h2 w2]));
	end
	
	% compute descriptors
	clear descs1 descs2;
	fprintf('Computing descriptors...\n'); tic;
	if ~use_seg
		descs1 = get_descriptors(im1,settings,1);
		descs2 = get_descriptors(im2,settings,1);
	else
		descs1 = get_descriptors_seg(im1,settings,1,[],[],single(seg1));
		descs2 = get_descriptors_seg(im2,settings,1,[],[],single(seg2));
	end
	s1 = size(descs1);
	s2 = size(descs2);
	descs1 = permute(reshape(descs1,s1(1)*s1(2)*s1(3),h1,w1),[2 3 1]);
	descs2 = permute(reshape(descs2,s2(1)*s2(2)*s2(3),h2,w2),[2 3 1]);
	fprintf('Done in %.2f\n',toc);

	% SIFT flow matching
	SIFTflowpara.alpha = 1;
	SIFTflowpara.d = 40;
	SIFTflowpara.gamma = 0.005;
	SIFTflowpara.nlevels = 4;
	SIFTflowpara.wsize = 5;
	SIFTflowpara.topwsize = 20;
	SIFTflowpara.nIterations = 60;
	
	fprintf('Computing SID-flow...\n');
	tic; [vx vy energy] = SIFTflowc2f(descs1,descs2,SIFTflowpara);
	fprintf('Done in %.2f\n',toc);

	% matching
	warp = warpImage(im2,vx,vy);

	% plot
	clear flow;
	flow(:,:,1) = vx;
	flow(:,:,2) = vy;
	figure;
	subplot(2,2,1); imshow(show_border_clr(im1,gt1,[1,0,0],2)); title('Image 1');
	subplot(2,2,2); imshow(show_border_clr(im2,gt2,[1,0,0],2)); title('Image 2');
	subplot(2,2,3); imshow(flowToColor(flow)); title(sprintf('%s: Flow I2>I1',label));
	subplot(2,2,4); imshow(show_border_clr(warp,gt1,[1,0,0],2)); title(sprintf('%s: Warp I2>I1',label));
	%imres = [show_border_clr(im1,gt1,[1,0,0],2) show_border_clr(im2,gt2,[1,0,0],2); im2double(flowToColor(flow)) show_border_clr(warp,gt1,[1,0,0],2)];
	%imwrite(imres,sprintf('%s-%d-to-1.png',label,k));
end