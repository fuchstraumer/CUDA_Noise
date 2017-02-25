#include "common\CommonInclude.h"
#include "common\CUDA_Include.h"
// Include a module
#include "modules\Base.h"
#include "modules\generators\Billow.h"
#include "modules\generators\FBM.h"
#include "modules\modifiers\Select.h"
// Include image writing class.
#include "image\Image.h"

using namespace noise::module;

int main() {
	static constexpr int img_size = 8192;
	FBM2D subject0(img_size, img_size, noise_t::PERLIN, 1.0f, 1.0f, 2342, 0.05f, 1.5f, 7, 1.5f);
	FBM2D subject1(img_size, img_size, noise_t::PERLIN, 1.0f, 1.0f, 22312451, 0.05f, 1.5f, 7, 1.5f);
	FBM2D selector(img_size, img_size, noise_t::PERLIN, 5.0f, 1.0f, 2411, 0.025f, 1.5f, 4, 1.5f);
	subject0.Generate();
	subject1.Generate();
	selector.Generate();
	subject0.SaveToPNG("subject0.png");
	subject1.SaveToPNG("subject1.png");
	selector.SaveToPNG("subject2.png");
	Select module(img_size, img_size, 0.3f, 0.6f, 0.0f, std::shared_ptr<Module>(&selector), std::shared_ptr<Module>(&subject0), std::shared_ptr<Module>(&subject1));
	module.Generate();
	module.SaveToPNG("output.png");
}