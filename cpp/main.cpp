#include "common\CommonInclude.h"
#include "common\CUDA_Include.h"

// Include a module
#include "modules\generators\Billow.h"
// Include image writing class.
#include "image\Image.h"
int main() {
	static constexpr int img_size = 2048;
	noise::module::Billow2D module(img_size, img_size, 1.0f, 1.0f, 28, 0.025f, 1.5f, 2, 0.5f);
	module.Generate();
	std::vector<float> test_data;
	test_data = module.GetGPUData();
	if (test_data.empty()) {
		std::cerr << "Test failed" << std::endl;
	}
	ImageWriter testWriter(img_size, img_size);
	testWriter.SetRawData(std::move(test_data));
	testWriter.ConvertRawData();
	testWriter.WritePNG("test.png");
}