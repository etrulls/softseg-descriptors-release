/////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it              //
// and/or modify it under the terms of the GNU General Public License  //
// version 2 (or higher) as published by the Free Software Foundation. //
//                                                                     //
// This program is distributed in the hope that it will be useful, but //
// WITHOUT ANY WARRANTY; without even the implied warranty of          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   //
// General Public License for more details.                            //
//                                                                     //
// Written and (C) by                                                  //
// Engin Tola                                                          //
//                                                                     //
// web   : http://cvlab.epfl.ch/~tola                                  //
// email : engin.tola@epfl.ch                                          //
//                                                                     //
// Adapted to meet the needs of:                                       //
// Iasonas Kokkinos                                                    //
// iasonas.kokkinos@ecp.fr	                                           //							
//                                                                     //
// Modified for segmentation-awareness by                              //
// Eduard Trulls                                                       //
// eduard.trulls@iri.upc.edu                                           //
//                                                                     //
// If you use this code for research purposes, please refer to the     //
// webpage above                                                       //
/////////////////////////////////////////////////////////////////////////

#include <mex.h>
#include <string.h> // Call to memset
#include <assert.h>
#include <cmath>

#define SIFT_TH 0.154
#define MAX_ITER 5

inline bool clip_vector( float* vec, int sz, float th );

inline void normalize_vector( float* vec, int hs );
inline void normalize_partial( float* desc, int gn, int hs );
inline void normalize_full( float* desc, int sz );
inline void normalize_sift( float* desc, int sz );

inline void u_compute_descriptor_00(const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out );
inline void u_compute_descriptor_01(const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out );
inline void u_compute_descriptor_10(const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out );
inline void u_compute_descriptor_11(const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out, int pos, int num_p );

inline void compute_descriptor( const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out, int pos, int num_p );

float *ev, *ev_out, *weights_out;
int num_ev, num_conv_ev;

// parameters
bool s_use_l1_distance;
int s_distance_type, s_masks_type;
float s_lambda;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if ((nrhs!=8)) {  printf("in nrhs != 8 \n"); return; }
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { printf("input 1 must be a single array\n"); return; } // H
    if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) { printf("input 2 must be a single array\n"); return; } // params
    if (mxGetClassID(prhs[2]) != mxSINGLE_CLASS) { printf("input 3 must be a single array\n"); return; } // grid
    if (mxGetClassID(prhs[3]) != mxSINGLE_CLASS) { printf("input 4 must be a single array\n"); return; } // ostable
    if (mxGetClassID(prhs[4]) != mxSINGLE_CLASS) { printf("input 5 must be a single array\n"); return; } // orientation
		if (mxGetClassID(prhs[7]) != mxSINGLE_CLASS) { printf("input 8 must be a single array\n"); return; } // eigenvectors
    bool do_subset = 0;
    float *X, *Y;
    
    do_subset  = 1;
    X = (float *)mxGetData(prhs[5]);
    Y = (float *)mxGetData(prhs[6]);

		int const num_dims_ev = mxGetNumberOfDimensions(prhs[7]);
    int const *dims_ev    = mxGetDimensions(prhs[7]);
		if( num_dims_ev != 3 && num_dims_ev != 4 )
		{ printf("input 8 must be HxWxN (no convolutions) or HxWxNxC (convolutions)\n"); return; } // H
		num_ev = (int) dims_ev[2];
		ev = (float*) mxGetData(prhs[7]);
		if( num_dims_ev == 4 )
			num_conv_ev = (int) dims_ev[3];
		else
			num_conv_ev = 1;
			
    float const *H     = (float *)mxGetData(prhs[0]);

		if( nlhs != 3 )
		{ printf("out nlhs != 3\n"); return; }
    
    // Histograms
    int const num_dims1 = mxGetNumberOfDimensions(prhs[0]);
    int const *dims1    = mxGetDimensions(prhs[0]);
    int hs = dims1[1];
    
    // params
    float const *params = (float *)mxGetData(prhs[1]);
    int const nd_params = mxGetNumberOfDimensions(prhs[1]);
    int const *nparams  = mxGetDimensions(prhs[1]);
    int DS = params[0];
#ifdef DEBUG
printf("----------------------------------------------\n");
printf("DS : %f\n", params[0] );
printf("HN : %f\n", params[1] );
printf("H  : %f\n", params[2] );
printf("W  : %f\n", params[3] );
printf("R  : %f\n", params[4] );
printf("RQ : %f\n", params[5] );
printf("TQ : %f\n", params[6] );
printf("HQ : %f\n", params[7] );
printf("SI : %f\n", params[8] );
printf("LI : %f\n", params[9] );
printf("NT : %f\n", params[10] );
printf("GOR: %f\n", params[11] );
printf("----------------------------------------------\n");
#endif

// get some parameters
if( nparams[1] != 15 )
{
	printf("Bad settings");
	return;
}
s_masks_type = (int) params[11];
s_use_l1_distance = (bool) params[12];
s_distance_type = (int) params[13];
s_lambda = (float) params[14];
char s[256];
sprintf(s, "Config: masks type: %d, l1 distance: %d, distance type: %d, lambda: %f\n", s_masks_type, s_use_l1_distance, s_distance_type, s_lambda);
printf("%s",s);

// grid info
float const *grid = (float *)mxGetData(prhs[2]);
int const nd_grid = mxGetNumberOfDimensions(prhs[2]);
int const *ngrid  = mxGetDimensions(prhs[2]);
int gn = ngrid[0];

// ostable info
float const *ostable = (float *)mxGetData(prhs[3]);

// orientation
float const *ori = (float *)mxGetData(prhs[4]);
if( *ori < 0 || *ori > 360 ) {
    printf("orientation %f must be [0,360)\n", *ori);
    return;
}

float shift = ostable[ (int)*ori ];

// output
int h=params[2];
int w=params[3];
int hw;
//if (do_subset) {
    hw = mxGetM(prhs[5]);
//}
//else {
//    hw = h*w;
//}

int odim[] = {DS, hw};
plhs[0] = mxCreateNumericArray(2, odim, mxSINGLE_CLASS, mxREAL);
float *desc_out  = (float *)mxGetData(plhs[0]);
memset( desc_out, 0, sizeof(float)*DS*hw );

// output, ev
int num_g = (int) params[1]; // grid size
int odim2[] = {num_g,hw,num_ev};
plhs[1] = mxCreateNumericArray(3, odim2, mxSINGLE_CLASS, mxREAL);
ev_out  = (float *)mxGetData(plhs[1]);
memset( ev_out, 0, sizeof(float)*num_g*hw*num_ev );

// output, weights
int odim3[] = {num_g,hw};
plhs[2] = mxCreateNumericArray(2, odim3, mxSINGLE_CLASS, mxREAL);
weights_out  = (float *)mxGetData(plhs[2]);
memset( weights_out, 0, sizeof(float)*num_g*hw );

for (int c = 0; c<hw; c++)
{
    int y = (int) Y[c];
    int x = (int) X[c];
		compute_descriptor(H, params, grid, y, x, shift, desc_out+(c)*DS, c, hw);
}
}

inline void compute_descriptor( const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out, int pos, int num_p ) 
{
    int si = params[8];
    int li = params[9];
    int nt = params[10];
    
    if     ( si == 0 && li == 0 ) u_compute_descriptor_00(H, params, grid, y, x, shift, desc_out);
    else if( si == 0 && li == 1 ) u_compute_descriptor_01(H, params, grid, y, x, shift, desc_out);
    else if( si == 1 && li == 0 ) u_compute_descriptor_10(H, params, grid, y, x, shift, desc_out);
    else if( si == 1 && li == 1 ) u_compute_descriptor_11(H, params, grid, y, x, shift, desc_out, pos, num_p);
    
    if( nt == 0 ) return;
    else if( nt == 1 ) normalize_partial(desc_out, params[1], params[7]);
    else if( nt == 2 ) normalize_full(desc_out, params[0] );
    else if( nt == 3 ) normalize_sift(desc_out, params[0] );
    else printf("\nunknown normalization\n");
}

inline void normalize_vector( float* hist, int hs ) 
{
    float s=0; int n = 0;
    for( int i=0; i<hs; i++ ) {s+= hist[i]*hist[i]; n = n + (int) (hist[i]!=0);}
    if( s!=0 ) {
        s = sqrt(s/float(n));
        for( int i=0; i<hs; i++ ) hist[i]/=s;
    }
}
inline void normalize_partial( float* desc, int gn, int hs ) 
{
    for( int g=0; g<gn; g++ )
        normalize_vector( desc+g*hs, hs );
}
inline void normalize_full( float* desc, int sz ) 
{
    normalize_vector(desc, sz);
}
inline bool clip_vector( float* vec, int sz, float th ) 
{
    bool retval=false;
    for( int i=0; i<sz; i++ )
        if( vec[i] > th ) {
        vec[i]=th;
        retval = true;
        }
    return retval;
}
inline void normalize_sift( float* desc, int sz ) 
{
    int iter=0;
    bool change=true;
    while( iter<MAX_ITER && change ) {
        normalize_vector(desc, sz);
        change = clip_vector(desc, sz, SIFT_TH);
        iter++;
    }
}

inline void u_compute_descriptor_00(const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out ) 
{
    int h=params[2];
    int w=params[3];
    int hw = h*w;
    int hq = params[7];
    int id, g, c, j;
    float* hist;
    const float* cube=0;
    int yy, xx, cnt;
    int hn = params[1];
    int ishift = (int)shift;

		mexErrMsgTxt("00 todo");

    for( g=0; g<hn; g++ ) {
        c  = grid[g];
        yy = round(y + grid[g+  hn]);
        xx = round(x + grid[g+2*hn]);
        if( 0 > yy || yy >= h || 0>xx || xx >= w ) continue;

        id = yy*w+xx;
        cube = H+(c-1)*hw*hq+id;
        
        hist = desc_out + g*hq;
        
        for( j=0; j<hq-ishift; j++ )
            hist[j] = cube[(j+ishift)*hw];
        for( cnt=0; cnt<ishift; cnt++, j++ )
            hist[j] = cube[cnt*hw];
    }
}
inline void u_compute_descriptor_01(const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out ) {
    int h=params[2];
    int w=params[3];
    int hw = h*w;
    int hq = params[7];
    int id, g, c, j;
    float* hist;
    const float* cube=0;
    int yy, xx, cnt;
    int hn = params[1];
    int ishift = (int)shift;
    float f = shift - ishift;

		mexErrMsgTxt("01 todo");

    for( g=0; g<hn; g++ ) {
        c  = grid[g];
        yy = floor(y + grid[g+  hn]);
        xx = floor(x + grid[g+2*hn]);
        if( 0 > yy || yy >= h || 0>xx || xx >= w ) continue;
        
        id = yy*w+xx;
        cube = H+(c-1)*hw*hq+id;
        
        hist = desc_out + g*hq;
        
        for( j=0; j<hq-ishift; j++ ) hist[j] = cube[(j+ishift)*hw];
        for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] = cube[cnt*hw];
        
        float tmp = hist[0];
        for( cnt=0; cnt<hq-1; cnt++ )
            hist[cnt] = f*hist[cnt+1]+(1-f)*hist[cnt];
        hist[hq-1] = f*tmp+(1-f)*hist[hq-1];
    }
}
inline void u_compute_descriptor_10(const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out ) {
    int h=params[2];
    int w=params[3];
    int hw = h*w;
    int hq = params[7];
    int g, c, j;
    float* hist;
    const float* cube=0;
    float yy, xx;
    int iy, ix;
    int cnt;
    int hn = params[1];
    int ishift = (int)shift;

		mexErrMsgTxt("10 todo");

    for( g=0; g<hn; g++ ) {
        c  = grid[g];
        yy = y + grid[g+  hn];
        xx = x + grid[g+2*hn];
        iy = (int)yy;
        ix = (int)xx;
        if( 0 > iy || iy >= h-1 || 0>ix || ix >= w-1 ) continue;
        
        float b = yy-iy;
        float a = xx-ix;
        
        hist = desc_out + g*hq;
        
        // A C
        // B D
        
        // A
        cube = H+(c-1)*hw*hq+iy*w+ix;
        for( j=0;     j<hq-ishift; j++    ) hist[j] = (1-a)*(1-b)*cube[(j+ishift)*hw];
        for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] = (1-a)*(1-b)*cube[cnt*hw];
        
        // B
        cube = H+(c-1)*hw*hq+iy*w+ix+w;
        for( j=0;     j<hq-ishift; j++    ) hist[j] += b*(1-a)*cube[(j+ishift)*hw];
        for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] += b*(1-a)*cube[cnt*hw];
        
        // C
        cube = H+(c-1)*hw*hq+iy*w+ix+1;
        for( j=0;     j<hq-ishift; j++    ) hist[j] += a*(1-b)*cube[(j+ishift)*hw];
        for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] += a*(1-b)*cube[cnt*hw];
        
        // D
        cube = H+(c-1)*hw*hq+iy*w+ix+w+1;
        for( j=0;     j<hq-ishift; j++    ) hist[j] += a*b*cube[(j+ishift)*hw];
        for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] += a*b*cube[cnt*hw];
    }
}
inline void u_compute_descriptor_11(const float* H, const float* params, const float* grid, float y, float x, float shift, float* desc_out, int pos, int num_p )
{
	int h=params[2];
	int w=params[3];
	int hw = h*w;
	int hq = params[7];
	int g, c, j;
	float* hist;
	const float* cube=0;
	float yy, xx;
	int iy, ix;
	int cnt;
	int hn = params[1];
	int ishift = floor(shift);
	float f=shift-ishift;
	float weight;

	int region_c = 0;
	float *ev_i;
	int n_rays = params[6];
	int n_scales = (int) (params[1] / n_rays); // quick fix
	float d;

	// i don't have segmentation data for the center point so i can't do anything here
	if( round(x)<0 || round(y) >= h || round(x)<0 || round(x) >= w )
		return;

	float ev_center[num_ev];
	for(int v=0; v<num_ev; v++)
		// always the first convolution (which is negligible)
		ev_center[v] = ev[((int)round(y))+((int)round(x))*h+v*hw];

	// weighting factor on modes 2-4
	float ev_center_factor[num_ev];
	if( s_distance_type == 2 || s_distance_type == 3 || s_distance_type == 4 )
	{
		for(int v=0; v<num_ev; v++)
		{
			if( s_distance_type == 2 ) // minimum distance
				ev_center_factor[v] = 1E6;
			else if( s_distance_type == 3 ) // average distance
				ev_center_factor[v] = 0;
			else // maximum distance
				ev_center_factor[v] = 0;

			// we check a few scales
			int num_scales_to_check = 3; // do not set this to 0
			for( int i_scale=0; i_scale<num_scales_to_check; i_scale++)
				for( int i_ray=0; i_ray<n_rays; i_ray++)
				{
					g = i_ray + i_scale*n_rays;
					yy = y + grid[g+  hn];
					xx = x + grid[g+2*hn];
					iy = floor(yy);
					ix = floor(xx);

					if( 0 > iy || iy >= h-1 || 0>ix || ix >= w-1 )
							continue;

					// first convolution only
					d = fabs( ev_center[v] - ev[((int)round(yy)) + ((int)round(xx))*h + v*hw] );
					if( s_distance_type == 2 )
					{
						if( d < ev_center_factor[v]  )
							ev_center_factor[v] = d;
					}
					else if( s_distance_type == 3 )
						ev_center_factor[v] += d / (n_rays * num_scales_to_check);
					else
					{
						if( d > ev_center_factor[v]  )
							ev_center_factor[v] = d;
					}
				}

			// enforce lower bound
			if( ev_center_factor[v] < 1E-6 )
				ev_center_factor[v] = 1E-6;
		}
	}
		
	for( int i_scale=0; i_scale<n_scales; i_scale++)
	{
		for( int i_ray=0; i_ray<n_rays; i_ray++) 
		{
			g = i_ray + i_scale*n_rays;

			c  = grid[g];
			yy = y + grid[g+  hn];
			xx = x + grid[g+2*hn];
			iy = floor(yy);
			ix = floor(xx);

			// if we're not using convolved maps of eigenvectors, set the convolution index to zero
			int ev_index = (num_conv_ev == 1) ? 0 : c-1;
      
			// compute weights from the eigenvectors
			// outside the image: large, artificial distance
			if( 0 > iy || iy >= h-1 || 0>ix || ix >= w-1 )
			{
				// skip outside the image, but first we set the distance between eigenvalues to a large value
				for(int v=0; v<num_ev; v++)
					//ev_out[g + pos*hn + v*num_p*hn] = s_use_l1_distance ? num_ev*range_ev : sqrt(num_ev)*range_ev;	
					ev_out[g + pos*hn + v*num_p*hn] = 1E6;				
				continue;
			}

			// compute weight for this grid point from the soft segmentation
			float k = 0;
			// average distances
			if( s_distance_type == 1 )
			{
				float dist;
				for(int v=0; v<num_ev; v++)
				{
					// DO NOT CALL ABS(): doesn't work (sets everything to 0. why?)
					dist = fabs( ev_center[v] - ev[((int)round(yy)) + ((int)round(xx))*h + v*hw + ev_index*num_ev*hw] );
					ev_out[g + pos*hn + v*num_p*hn] = dist;
				}
				
				// combine distances
				if( s_use_l1_distance )
				{
					for(int v=0; v<num_ev; v++)
						k += ev_out[g + pos*hn + v*num_p*hn];
					k = k / num_ev; // reweight by number of vectors
				}
				else
				{
					for(int v=0; v<num_ev; v++)
						k += ev_out[g + pos*hn + v*num_p*hn] * ev_out[g + pos*hn + v*num_p*hn];
					k = sqrt(k / num_ev); // reweight by number of vectors
				}
			}
			// experiment based on iasonas' suggestion
			else if( s_distance_type == 2 || s_distance_type == 3 || s_distance_type == 4 )
			{
				float dist;
				for(int v=0; v<num_ev; v++)
				{
					// DO NOT CALL ABS(): doesn't work (sets everything to 0. why?)
					dist = fabs( ev_center[v] - ev[((int)round(yy)) + ((int)round(xx))*h + v*hw + ev_index*num_ev*hw] );
					ev_out[g + pos*hn + v*num_p*hn] = dist / ev_center_factor[v]; // APPLY FACTOR
					//printf("factor: %f\n",ev_center_factor[v]);
				}
				
				// combine distances
				if( s_use_l1_distance )
				{
					for(int v=0; v<num_ev; v++)
						k += ev_out[g + pos*hn + v*num_p*hn];
					k = k / num_ev; // reweight by number of vectors
				}
				else
				{
					for(int v=0; v<num_ev; v++)
						k += ev_out[g + pos*hn + v*num_p*hn] * ev_out[g + pos*hn + v*num_p*hn];
					k = sqrt(k / num_ev); // reweight by number of vectors
				}
			}
			// accumulated (per ray)
			else if( s_distance_type == 5 )
			{
				float dist;
				for(int v=0; v<num_ev; v++)
				{
					// let's not interpolate here
					// first step: grid center to first ray point
					dist = fabs( ev_center[v] - ev[((int)round(y + grid[i_ray + hn])) + ((int)round(x + grid[i_ray + 2*hn]))*h + v*hw + ev_index*num_ev*hw] );
					// step until grid point
					for(int j_scale = 1; j_scale <= i_scale; j_scale++)
						dist += fabs( ev_center[v] - ev[((int)round(y + grid[i_ray + (j_scale-1)*n_rays + hn])) + ((int)round(x + grid[i_ray + j_scale*n_rays + 2*hn]))*h + v*hw + ev_index*num_ev*hw] );

					ev_out[g + pos*hn + v*num_p*hn] = dist;
				}
				
				// combine distances
				if( s_use_l1_distance )
				{
					for(int v=0; v<num_ev; v++)
						k += ev_out[g + pos*hn + v*num_p*hn];
					k = k / num_ev; // reweight by number of vectors
				}
				else
				{
					for(int v=0; v<num_ev; v++)
						k += ev_out[g + pos*hn + v*num_p*hn] * ev_out[g + pos*hn + v*num_p*hn];
					k = k / sqrt(num_ev); // reweight by number of vectors
				}
			}
			else
			{
				printf("Bad distance mode\n");
				return;
			}
			
			// BUILD MASKS
			// EXPONENTIAL
			if( s_masks_type == 1 )
				weight = exp(-s_lambda*k);
			// STEP FUNCTION
			else if( s_masks_type == 2 )
			{
				if( k <= s_lambda )
					weight = 1;
				else
					weight = 0;
			}
			// SIGMOID
			else if( s_masks_type == 3 )
			{
				// lower bound for th2; may want to change this for a sharper decline
				float th2 = 6.9 / s_lambda;
				weight = 1 - 1. / (1 + exp(-th2*(k - s_lambda)));
			}
			else
			{
				printf("Bad mask mode\n");
				return;
			}
			
			// get a copy
			weights_out[g + pos*hn] = weight;
      
      float b = yy-iy;
      float a = xx-ix;
      
      hist = desc_out + g*hq;
      
			// A C
      // B D
			
      // A
      cube = H+(c-1)*hw*hq+iy*w+ix;
      for( j=0;     j<hq-ishift; j++    ) hist[j] = (1-a)*(1-b)*cube[(j+ishift)*hw];
      for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] = (1-a)*(1-b)*cube[cnt*hw];
     
      // B
      cube = H+(c-1)*hw*hq+iy*w+ix+w;
      for( j=0;     j<hq-ishift; j++    ) hist[j] += b*(1-a)*cube[(j+ishift)*hw];
      for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] += b*(1-a)*cube[cnt*hw];
     
      // C
      cube = H+(c-1)*hw*hq+iy*w+ix+1;
      for( j=0;     j<hq-ishift; j++    ) hist[j] += a*(1-b)*cube[(j+ishift)*hw];
      for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] += a*(1-b)*cube[cnt*hw];
     
      // D
      cube = H+(c-1)*hw*hq+iy*w+ix+w+1;
      for( j=0;     j<hq-ishift; j++    ) hist[j] += a*b*cube[(j+ishift)*hw];
      for( cnt=0; cnt<ishift; cnt++, j++ ) hist[j] += a*b*cube[cnt*hw];
      
      float tmp = hist[0];
      for( cnt=0; cnt<hq-1; cnt++ )
          hist[cnt] = f*hist[cnt+1]+(1-f)*hist[cnt];
      hist[hq-1] = f*tmp+(1-f)*hist[hq-1];

			// apply weights
			for( j=0; j<hq; j++ )
				hist[j] *= weight;

		}
  }
}

