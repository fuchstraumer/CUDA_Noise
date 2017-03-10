// Include modules
#define USING_CNOISE_NAMESPACES
#include "modules\Modules.h"

inline void PrintMemInfo() {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cerr << " Total allocated: " << total - free << std::endl;
}

int main() {

	int img_size_x = 2048;
	int img_size_y = 1024;
	
	Billow3D test3d(img_size_x, img_size_y, 100, 0.0f, 0.0f, 0.0f, 314513412, 0.0025f, 1.8f, 8, 0.75f);
	test3d.Generate();
	for (int i = 0; i < 100; ++i) {
		char buff[32];
		sprintf(buff, "./img/3d/layer_%d.png", i);
		test3d.SaveToPNG(buff, i);
	}
}