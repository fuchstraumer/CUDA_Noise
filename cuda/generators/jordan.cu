#include "jordan.cuh"

__device__ float jordan_simplex(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	float2 deriv;
	float n = simplex2d(point.x * freq, point.x * freq, init_seed, &deriv);
	float sum = n * n;
	float2 dsum_warp = 0.40f * deriv;
	float2 dsum_damp = 1.0f * deriv;

	float amplitude = 0.80f;
	float damped_amplitude = amplitude * 0.50f;
	point.x *= freq;
	point.y *= freq;
	for (int i = 1; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		n = perlin2d(point.x + dsum_warp.x, point.x + dsum_warp.y, seed, &deriv);
		deriv *= n;
		n *= n;
		sum += damped_amplitude * n;
		dsum_warp += 0.20f * deriv;
		dsum_damp += 0.80f * deriv;
		point.x *= lacun;
		point.y *= lacun;
		amplitude *= persist;
		damped_amplitude = amplitude * (1 - (1.0f / (1.0f + dot(dsum_damp, dsum_damp))));
	}
	return sum;
}

__device__ float jordan_perlin(float px, float py, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
	float amplitude = 1.0f;
	// Scale point by freq
	float2 point = make_float2(px * freq, py * freq);
	float warp = 0.01f;
	// TODO: Seeding the function is currently pointless and doesn't actually do anything.
	// Use loop for octav-ing
	float result = 0.0f;
	float dx_sum = 0.0f, dy_sum = 0.0f;
	for (size_t i = 0; i < octaves; ++i) {
		int seed = (init_seed + i) & 0xffffffff;
		point.x += (warp * dx_sum);
		point.y += (warp * dy_sum);
		float2 dx_dy;
		float n = perlin2d(point.x, point.y, seed, &dx_dy);
		result += (1.0f - fabsf(n)) * amplitude;
		dx_sum += amplitude * dx_dy.x * -n;
		dy_sum += amplitude * dx_dy.y * -n;
		point.x *= lacun;
		point.y *= lacun;
		// Modify vars for next octave.
		amplitude *= persist * __saturatef(result);
	}
	return result;
}