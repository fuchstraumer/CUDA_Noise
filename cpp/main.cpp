#include "common\CUDA_Include.h"
// Include modules
#include "modules\Base.h"
#include "modules\generators\Billow.h"
#include "modules\generators\RidgedMulti.h"
#include "modules\generators\FBM.h"

using namespace noise::module;

int main() {
	static constexpr int img_size = 1024;
	for (size_t oct = 1; oct < 11; ++oct) {
		Billow2D test0(img_size, img_size, noise_t::SIMPLEX, 1, 1, 34567, 0.001f, 1.5f, oct, 0.9f);
		test0.Generate();
		char fname[64];
		sprintf(fname, "billow_simplex_curve5_octave%d.png", oct);
		test0.SaveToPNG(fname);
	}
	std::cerr << "Simplex finished, moving on to perlin." << std::endl;
	for (size_t oct = 1; oct < 11; ++oct) {
		Billow2D test1(img_size, img_size, noise_t::PERLIN, 1, 1, 1486, 0.001f, 1.5f, oct, 0.9f);
		test1.Generate();
		char fname[64];
		sprintf(fname, "billow_perlin_curve5_octave%d.png", oct);
		test1.SaveToPNG(fname);
	}
}