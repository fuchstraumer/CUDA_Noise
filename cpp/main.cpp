#include "common\CUDA_Include.h"
// Include modules
#include "modules\Base.h"
#include "modules\generators\Billow.h"
#include "modules\generators\RidgedMulti.h"
#include "modules\generators\FBM.h"
#include "modules\modifiers\Select.h"
#include "modules\generators\PerlinTex.h"
using namespace noise::module;

int main() {
	static int img_size = 4096;
	RidgedMulti test0(img_size, img_size, noise_t::PERLIN, 0.0f, 0.0f, 34567, 0.001f, 1.6f, 12, 0.6f);
	test0.Generate();
	test0.SaveToPNG("perlin_regular.png");
	PerlinTexBase texTest(img_size, img_size, make_float2(0.0f, 0.0f), 2131231);
	texTest.Generate();
	texTest.SaveToPNG("perlin_texture_LUT.png");
}