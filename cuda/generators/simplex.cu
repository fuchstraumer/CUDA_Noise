#include "simplex.cuh"


/*

	2D Simplex noise, from: https://github.com/Auburns/FastNoiseSIMD/blob/master/FastNoiseSIMD/FastNoiseSIMD_internal.cpp 
	& https://github.com/Auburns/FastNoise/blob/master/FastNoise.cpp
	Ported to CUDA, but credit goes to original creator. Used mix of SIMD code to get ideas for seeding without any 
	run-time generated LUTs

*/

// Following is WIP/abandoned simplex implementation based on FastNoise. Went with Stefan Gustavsons stuff because its
// generally easier to work with, especially because it includes derivatives we can find if we like.
/*__device__ __constant__ short GRAD_X[12] =
{
	1,-1, 1,-1,
	1,-1, 1,-1,
	0, 0, 0, 0
};
__device__ __constant__ short GRAD_Y[12] =
{
	1, 1,-1,-1,
	0, 0, 0, 0,
	1,-1, 1 -1
};
__device__ __constant__ short GRAD_Z[12] =
{
	0, 0, 0, 0,
	1, 1,-1,-1,
	1, 1,-1 -1
};

__device__ __constant__ short GRAD_4D[128] =
{
	0,1,1,1,0,1,1,-1,0,1,-1,1,0,1,-1,-1,
	0,-1,1,1,0,-1,1,-1,0,-1,-1,1,0,-1,-1,-1,
	1,0,1,1,1,0,1,-1,1,0,-1,1,1,0,-1,-1,
	-1,0,1,1,-1,0,1,-1,-1,0,-1,1,-1,0,-1,-1,
	1,1,0,1,1,1,0,-1,1,-1,0,1,1,-1,0,-1,
	-1,1,0,1,-1,1,0,-1,-1,-1,0,1,-1,-1,0,-1,
	1,1,1,0,1,1,-1,0,1,-1,1,0,1,-1,-1,0,
	-1,1,1,0,-1,1,-1,0,-1,-1,1,0,-1,-1,-1,0
};

__device__ __constant__ float F3 = 1.0f / 3.0f;
__device__ __constant__ float G3 = 1.0f / 6.0f;

__device__ float simplex_single(const unsigned char offset, const float3 p) {
	// Following is messy, but best to avoid creating any local variables that we won't 
	// re-use constantly. (and even then, its still best to just avoid local vars)
	int3 ic = make_int3(
		floorf((p.x + p.y + p.z) * F3 + p.x), // Skew x-coord and get integral component. (thus, "ic")
		floorf((p.x + p.y + p.z) * F3 + p.y), // Y-coord skew
		floorf((p.x + p.y + p.y) * F3 + p.z)); // Z-coord skew

	// Same thing again: messy, but most simplex algorithms create three floats that are
	// used once to do this. We can't afford that much register space.
	// Getting fractional component here. (fc = fractional_component)
	float3 fc = make_float3(
		ic.x - ((ic.x + ic.y + ic.z) * G3),
		ic.y - ((ic.x + ic.y + ic.z) * G3),
		ic.z - ((ic.x + ic.y + ic.z) * G3)
	);

	// Construct simplex cell coords.
	short3 sc0, sc1;
	if (fc.x >= fc.y) {
		if (fc.y >= fc.z) {
			sc0 = make_short3(1, 0, 0);
			sc1 = make_short3(1, 1, 0);
		}
		else if (fc.x >= fc.z) {
			sc0
		}
	}
}*/

__device__ __constant__ unsigned char perm[512] = { 151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

__device__ __constant__ float grad2LUT[8][2] = {
	{ -1.0f, -1.0f },{ 1.0f,  0.0f },{ -1.0f, 0.0f },{ 1.0f,  1.0f } ,
	{ -1.0f,  1.0f },{ 0.0f, -1.0f },{ 0.0f,  1.0f },{ 1.0f, -1.0f }
};

__device__ void grad2(const int hash, float *gx, float *gy) {
	int h = hash & 7;
	*gx = grad2LUT[h][0];
	*gy = grad2LUT[h][1];
	return;
};

__device__ float curve5(const float a) {
	return (6.0f * a * a * a * a * a) - (15.0f * a * a * a * a) + (10.0f * a * a * a);
}

__device__ float simplex2d(const float2 p, float2 *dnoise){
	float n0, n1, n2; /* Noise contributions from the three simplex corners */
	float gx0, gy0, gx1, gy1, gx2, gy2; /* Gradients at simplex corners */
	float t0, t1, t2, x1, x2, y1, y2;
	static float F2 = 0.366025403f;
	static float G2 = 0.211324865f;


	// Integral component. USed to get fractional component (fc)
	int2 ic = make_int2(
		floorf(p.x + ((p.x + p.y) * F2)),
		floorf(p.y + ((p.x + p.y) * F2)));

	// Get fractional component, which gives x,y distances from simplex cell's origin.
	float2 fc = make_float2(
		p.x - (ic.x - (ic.x + ic.y) * G2),
		p.y - (ic.y - (ic.x + ic.y) * G2)
	);

	/* For the 2D case, the simplex shape is an equilateral triangle.
	* Determine which simplex we are in. */
	short i1, j1; /* Offsets for second (middle) corner of simplex in (i,j) coords */
	if (fc.x > fc.y) { 
		/* lower triangle, XY order: (0,0)->(1,0)->(1,1) */
		i1 = 1; 
		j1 = 0; 
	} 
	else {
		/* upper triangle, YX order: (0,0)->(0,1)->(1,1) */
		i1 = 0; 
		j1 = 1; 
	}     


	x1 = fc.x - i1 + G2; /* Offsets for middle corner in (x,y) unskewed coords */
	y1 = fc.y - j1 + G2;
	x2 = fc.x - 1.0f + 2.0f * G2; /* Offsets for last corner in (x,y) unskewed coords */
	y2 = fc.y - 1.0f + 2.0f * G2;

	/* Wrap the integer indices at 256, to avoid indexing perm[] out of bounds */
	ic.x = ic.x % 256;
	ic.y = ic.y % 256;

	/* Calculate the contribution from the three corners */
	t0 = 0.5f - fc.x * fc.x - fc.y * fc.y;
	if (t0 < 0.0f) {
		t0 = n0 = gx0 = gy0 = 0.0f; /* No influence */
	}
	else {
		grad2(perm[ic.x + perm[ic.y]], &gx0, &gy0);
		t0 = t0 * t0;
		t0 = t0 * t0;
		n0 = t0 * (gx0 * fc.x + gy0 * fc.y);
	}

	t1 = 0.5f - x1 * x1 - y1 * y1;
	if (t1 < 0.0f) {
		t1 = n1 = gx1 = gy1 = 0.0f; /* No influence */
	}
	else {
		grad2(perm[ic.x + i1 + perm[ic.y + j1]], &gx1, &gy1);
		t1 = t1 * t1;
		t1 = t1 * t1;
		n1 = t1 * (gx1 * x1 + gy1 * y1);
	}

	t2 = 0.5f - x2 * x2 - y2 * y2;
	if (t2 < 0.0f) {
		t2 = n2 = gx2 = gy2 = 0.0f; /* No influence */
	}
	else {
		grad2(perm[ic.x + 1 + perm[ic.y + 1]], &gx2, &gy2);
		t2 = t2 * t2;
		t2 = t2 * t2;
		n2 = t2 * (gx2 * x2 + gy2 * y2);
	}

	/* Compute derivative, if requested by supplying non-null pointers
	* for the last two arguments */
	if (dnoise != nullptr) {
		dnoise->x = sqrtf(t0) * sqrtf(sqrtf(t0)) * (gx0 * fc.x + gy0 * fc.y) * fc.x;
		dnoise->y = sqrtf(t0) * sqrtf(sqrtf(t0)) * (gx0 * fc.x + gy0 * fc.y) * fc.y;
		dnoise->x += sqrtf(t1) * sqrtf(sqrtf(t1)) * (gx1 * x1 + gy1 * y1) * x1;
		dnoise->y += sqrtf(t1) * sqrtf(sqrtf(t1)) * (gx1 * x1 + gy1 * y1) * y1;
		dnoise->x += sqrtf(t2) * sqrtf(sqrtf(t2)) * (gx2* x2 + gy2 * y2) * x2;
		dnoise->y += sqrtf(t2) * sqrtf(sqrtf(t2)) * (gx2* x2 + gy2 * y2) * y2;
		dnoise->x *= -8.0f;
		dnoise->y *= -8.0f;
		dnoise->x += t0 * gx0 + t1 * gx1 + t1 * gx2;
		dnoise->y += t0 * gy0 + t1 * gy1 + t2 * gy2;
		dnoise->x *= 40.0f; /* Scale derivative to match the noise scaling */
		dnoise->y *= 40.0f;
	}
	return 40.0f * (n0 + n1 + n2);
}