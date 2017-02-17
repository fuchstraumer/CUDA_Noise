#include "perlin.cuh"

#ifndef HALF_PRECISION_SUPPORT

__device__ float perlin2d(cudaTextureObject_t perm_tex, cudaTextureObject_t grad_tex, float2 point, int seed) {
	
	point.x = point.x * 0.01f;
	point.y = point.y * 0.01f;

	// Calculate 2D integer coordinates and fractional component 
	float2 i = make_float2(floorf(point.x), floorf(point.y));
	float2 f = make_float2(point.x - i.x, point.y - i.y);

	// Get weights.
	float2 w;
	w.x = f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f);
	w.y = f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f);
	float4 w4 = make_float4(1.0f, w.x, w.y, w.x * w.y);

	// Get four randomly permutated indices from the noise lattice nearest "point"
	// and offset them by the seed.
	uchar4 tmp = tex2D<uchar4>(perm_tex, i.x / 256 + 0.50f, i.y / 256 + 0.50f);
	float4 perm = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
	perm = perm + seed;

	// Permute the fourst indices again and get the 2D gradient for each of
	// the four new coord-seed pairs.
	float4 gLeft, gRight;
	uchar4 tmp0 = tex2D<uchar4>(grad_tex, perm.x + 0.50f, perm.y + 0.50f);
	gLeft = make_float4(tmp0.x, tmp0.y, tmp0.z, tmp0.w);
	gLeft = gLeft * 2.0f;
	gLeft = gLeft - 1.0f;
	uchar4 tmp1 = tex2D<uchar4>(grad_tex, perm.z + 0.50f, perm.w + 0.50f);
	gRight = make_float4(tmp1.x, tmp1.y, tmp1.z, tmp1.w);
	gRight = gRight * 2.0f;
	gRight = gRight - 1.0f;

	// Evaluate gradients at four lattice points.
	float nLeftTop = dot(make_float2(gLeft.x, gLeft.y), f);
	float nRightTop = dot(make_float2(gRight.x, gRight.y), f + make_float2(-1.0f, 0.0f));
	float nLeftBottom = dot(make_float2(gLeft.z, gLeft.w), f + make_float2(0.0f, -1.0f));
	float nRightBottom = dot(make_float2(gRight.z, gRight.w), f + make_float2(-1.0f, -1.0f));

	// Blend gradients.
	float4 gradientBlend = make_float4(nLeftTop, nRightTop - nLeftTop, nLeftBottom - nLeftTop,
		nLeftTop - nRightTop - nLeftBottom + nRightBottom);
	float n = dot(gradientBlend, w4);

	// Return value.
	return n * 1.530734f;
}

#else

// TODO: Removed these until its re-implemented. Need to figure out how it works with textures.

#endif // !HALF_PRECISION_SUPPORT
