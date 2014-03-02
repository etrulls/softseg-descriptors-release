%% dependencies
clear all
cd ~/dev/newsseg/release
addpath ~/dev/vlfeat/toolbox
addpath sid
addpath gbd
addpath sift-flow
addpath sift-flow/mex
vl_setup('quiet');

%% settings
f1 = 0.5; % resize factor, image 1
f2 = f1; % resize factor, image 2
settings.lambda = 40;
settings.scale = 8; % size of one SIFT bin (integer)
settings.norm_mask = 1; % normalize the mask

%% sift-flow
SIFTflowpara.alpha=1;
SIFTflowpara.d=40;
SIFTflowpara.gamma=0.005;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=5;
SIFTflowpara.topwsize=20;
SIFTflowpara.nIterations=60;

%% get images
im1 = im2double(imread('data/cars3/cars3_01.jpg'));
im2 = im2double(imread('data/cars3/cars3_10.jpg'));
seg1 = softSegs(im1);
seg2 = softSegs(im2);
gt1 = im2double(imread('data/cars3/cars3_01.pgm'));
gt2 = im2double(imread('data/cars3/cars3_10.pgm'));

im1 = imresize(im1,f1);
seg1 = imresize(seg1,f1);
im2 = imresize(im2,f2);
seg2 = imresize(seg2,f2);
gt1 = imresize(gt1,f1,'nearest');
gt2 = imresize(gt2,f2,'nearest');
[h1,w1,n] = size(seg1);
[h2,w2,n] = size(seg2);

im1 = clip(im1); im2 = clip(im2);
seg1 = clip(seg1); seg2 = clip(seg2);

%% sift-flow with SIFT
fprintf('(1/2) OPTICAL FLOW WITH DSIFT DESCRIPTORS\n');
descs1 = sdsift(im1,[],settings);
descs2 = sdsift(im2,[],settings);

tic; [vx,vy,energy] = SIFTflowc2f(descs1,descs2,SIFTflowpara);
fprintf('Computed SIFT-flow in %.2f\n',toc);
warp = warpImage(im2,vx,vy);
clear flow;
flow(:,:,1) = vx;
flow(:,:,2) = vy;
figure; imshow([show_border_clr(im1,gt1,[1,0,0],2) show_border_clr(im2,gt2,[1,0,0],2); im2double(flowToColor(flow)) show_border_clr(warp,gt1,[1,0,0],2)]);
title('DSIFT-flow');

%% sift-flow with segmentation-aware SIFT
fprintf('(2/2) OPTICAL FLOW WITH SDSIFT DESCRIPTORS\n');
[descs1,masks1] = sdsift(im1,seg1,settings);
[descs2,masks2] = sdsift(im2,seg2,settings);

m = reshape(masks1,h1*w1,16);
diff1 = std(m'); % stdev over the 16 mask values (higher means more background interference)
diff1 = (max(diff1(:))-diff1(:))/(max(diff1(:))-min(diff1(:))); % normalize and invert
diff1 = reshape(diff1,h1,w1);
m = reshape(masks2,h2*w2,16);
diff2 = std(m'); % stdev over the 16 mask values (higher means more background interference)
diff2 = (max(diff2(:))-diff2(:))/(max(diff2(:))-min(diff2(:))); % normalize and invert
diff2 = reshape(diff2,h2,w2);

tic; [vx,vy,energy] = SIFTflowc2f(descs1,descs2,SIFTflowpara);
fprintf('Computed SIFT-flow in %.2f\n',toc);
warp = warpImage(im2,vx,vy);
clear flow;
flow(:,:,1) = vx;
flow(:,:,2) = vy;
figure; imshow([seg1(:,:,1:3) seg2(:,:,1:3); repmat(diff1,[1 1 3]) repmat(diff2,[1 1 3])]);
title('Soft segmentation / Mask magnitude');
figure; imshow([show_border_clr(im1,gt1,[1,0,0],2) show_border_clr(im2,gt2,[1,0,0],2); im2double(flowToColor(flow)) show_border_clr(warp,gt1,[1,0,0],2)]);
title('SDSIFT-flow');
