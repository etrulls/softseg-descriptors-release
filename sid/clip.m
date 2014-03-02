function im2 = clip(im1)
	im2 = im1;
	im2(im2<0) = 0;
	im2(im2>1) = 1;
end