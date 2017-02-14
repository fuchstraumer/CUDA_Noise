#include "perlin.cuh"

#ifndef HALF_PRECISION_SUPPORT

__device__ float perlin2d(cudaTextureObject_t perm_tex, cudaTextureObject_t grad_tex, float2 point, int seed) {
	
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
	float4 perm = tex2D<float4>(perm_tex, i.x / 256, i.y / 256);
	perm = perm + seed;

	// Permute the fourst indices again and get the 2D gradient for each of
	// the four new coord-seed pairs.
	float4 g1, g2;
	g1 = tex2D<float4>(grad_tex, perm.x, perm.y);
	g1 = g1 * 2.0f;
	g1 = g1 - 1.0f;
	g2 = tex2D<float4>(grad_tex, perm.z, perm.w);
	g2 = g2 * 2.0f;
	g2 = g2 - 1.0f;

	// Evaluate gradients at four lattice points.
	float a, b, c, d;
	a = dot(make_float2(g1.x, g1.y), f);
	b = dot(make_float2(g2.x, g2.y), f + make_float2(-1.0f, 0.0f));
	c = dot(make_float2(g1.z, g1.w), f + make_float2(0.0f, -1.0f));
	d = dot(make_float2(g2.z, g2.w), f + make_float2(-1.0f, -1.0f));

	// Blend gradients.
	float4 grads = make_float4(a, b - a, c - 1, a - b - c + d);
	float n = dot(grads, w4);

	// Return value.
	return n * 1.50f;
}

#else

// TODO: Removed these until its re-implemented. Need to figure out how it works with textures.

#endif // !HALF_PRECISION_SUPPORT
