// Include modules
#define USING_CNOISE_NAMESPACES
#include "modules\Modules.h"

inline void PrintMemInfo() {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cerr << " Total allocated: " << total - free << std::endl;
}

int main() {

	constexpr int img_size_x = 512;
	constexpr int img_size_y = 512;
	constexpr float sea_level = 0.10f;
	constexpr float continent_freq = 0.000125f / 4;
	constexpr float continent_lacun = 2.10f;

	FBM2D img(img_size_x, img_size_x, noise_t::PERLIN, 0.111f, 0.002f, 12412321, 0.009f, 2.2f, 11, 0.7f);
	img.Generate();
	img.SaveToPNG_16("high_depth.png");

}