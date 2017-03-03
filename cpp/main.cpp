#include "common\CUDA_Include.h"
// Include modules
#include "modules\Base.h"
#include "modules\generators\Billow.h"
#include "modules\generators\RidgedMulti.h"
#include "modules\generators\FBM.h"
#include "modules\modifiers\Select.h"
#include "modules\generators\DecarpientierSwiss.h"
using namespace noise::module;

int main() {
	static int img_size = 8192;
	int i = 0;
	DecarpientierSwiss ptb(img_size, img_size, noise_t::PERLIN, 0.12232f, -1.232f, 32341, 0.001f, 1.85f, 10, 0.95f);
	ptb.Generate();
	ptb.SaveToPNG("dc_swiss_perlin.png");
}