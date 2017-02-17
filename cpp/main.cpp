#include "stdafx.h"
#include "modules\Billow.h"
#include "image\Image.h"
int main() {
	noise::module::Billow2D module(512, 512, 1.0f, 1.0f, 2495, 6.5f, 2.0f, 5, 0.75f);
	module.Generate();
	std::vector<float> test_data;
	test_data = module.GetGPUData();
	if (test_data.empty()) {
		std::cerr << "Test failed" << std::endl;
	}
	ImageWriter testWriter(512, 512);
	testWriter.SetRawData(std::move(test_data));
	testWriter.ConvertRawData();
	testWriter.WritePNG("test.png");
}