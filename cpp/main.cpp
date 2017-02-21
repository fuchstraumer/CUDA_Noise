#include "common\CommonInclude.h"
#include "common\CUDA_Include.h"

// Include a module
#include "modules\generators\Billow.h"
// Include image writing class.
#include "image\Image.h"
int main() {

	noise::module::Billow2DSimplex module(4096, 4096, 1.0f, 1.0f, 2, 0.07f, 2.0f, 2);
	module.Generate();
	std::vector<float> test_data;
	test_data = module.GetGPUData();
	if (test_data.empty()) {
		std::cerr << "Test failed" << std::endl;
	}
	ImageWriter testWriter(4096, 4096);
	testWriter.SetRawData(std::move(test_data));
	testWriter.ConvertRawData();
	testWriter.WritePNG("test.png");
}