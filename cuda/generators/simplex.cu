#include "simplex.cuh"

/*
	Ported from Stefan Gustavson's simplex noise code
*/

__device__ float2 grad2(int hash, cudaTextureObject_t grad) {
	int h = hash & 7;
	float x, y;
	x = tex2D<float>(grad, h + 0.50f, 1.50f);
	y = tex2D<float>(grad, h + 0.50f, 0.50f);
	return make_float2(x, y);
}

__device__ float simplex2d(cudaTextureObject_t perm, cudaTextureObject_t grad, float2 point, int seed){
	// Noise contributions from three simplex corners
	float n0, n1, n2;
	// Gradients at simplex corners.
	float2 g0, g1, g2;
	// Other various temp vars
	float t0, t1, t2, x1, x2, y1, y2;
	float t20, t40, t21, t41, t22, t42;

	// Skew input space and determine what simplex cell we're in.
	float s = (point.x + point.y) * 0.366025403f;
	float xs = point.x + s;
	float ys = point.y + s;
	int i = floorf(xs);
	int ii = i;
	int j = floorf(ys);
	int jj = j;

	// Unskew cell back to XY space
	float t = i + j * 0.211324865f;
	float X0 = i - t;
	float Y0 = j - t;
	// XY distances from cell origin.
	float x0 = point.x - X0;
	float y0 = point.y - Y0;

	// Determine which simplex we're in.
	int i1, j1;
	if (x0 > y0) {
		i1 = 1;
		j1 = 0;
	}
	else {
		i1 = 0;
		j1 = 1;
	}

	// Perform step offsets, from XY space to IJ (simplex) space
	x1 = x0 - i1 + 0.211324865f;
	y1 = y0 - j1 + 0.211324865f;
	x2 = x0 - 1.0f + 2.0f * 0.211324865f;
	y2 = y0 - 1.0f + 2.0f * 0.211324865f;

	// Wrap indices at 256
	ii = i % 256;
	jj = j % 256;

	// Calculate contribution from three corners of the simplex now.
	
	// First corner.
	t0 = 0.5f - x0 * x0 - y0 * y0;
	if (t0 < 0.0f) {
		// This corner of the simplex has no influence.
		t40 = t20 = t0 = n0 = 0.0f;
		g0 = make_float2(0.0f, 0.0f);
	}
	else {
		unsigned char hash = tex1D<unsigned char>(perm, jj + 0.50f);
		hash = tex1D<unsigned char>(perm, ii + hash + 0.50f);
		g0 = grad2(hash, grad);
		t20 = t0 * t0;
		t40 = t20 * t20;
		n0 = t40 * (g0.x * x0 + g0.y * y0);
	}

	// Second corner.
	t1 = 0.5f - x1 * x1 - y1 * y1;
	if (t1 < 0.0f) {
		// This corner has no influence.
		t41 = t21 = t1 = n1 = 0.0f;
		g1 = make_float2(0.0f, 0.0f);
	}
	else {
		unsigned char h0 = tex1D<unsigned char>(perm, jj + j1 + 0.50f);
		unsigned char h1 = tex1D<unsigned char>(perm, h0 + ii + i1 + 0.50f);
		g1 = grad2(h1, grad);
		t21 = t1 * t1;
		t41 = t21 * t21;
		n1 = t41 * (g1.x * x1 + g1.y * y1);
	}

	// Third corner.
	t2 = 0.5f - x2 * x2 - y2 * y2;
	if (t2 < 0.0f) {
		// This corner has no influence.
		t42 = t22 = t2 = n2 = 0.0f;
		g2 = make_float2(0.0f, 0.0f);
	}
	else {
		unsigned char h0 = tex1D<unsigned char>(perm, jj + 1 + 0.50f);
		unsigned char h1 = tex1D<unsigned char>(perm, h0 + jj + 1 + 0.50f);
		g2 = grad2(h1, grad);
		t22 = t2 * t2;
		t42 = t22 * t22;
		n2 = t42 * (g2.x * x2 + g2.y * y2);
	}

	// Get final result
	float result = 40.0f * (n0 + n1 + n2);

	return result;
}
