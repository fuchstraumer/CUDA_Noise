#include "../common/CUDA_Include.h"
// Include modules
#include "modules/Base.h"
#include "modules/generators/Billow.h"
#include "modules\generators\RidgedMulti.h"
#include "modules\generators\FBM.h"
#include "modules\modifiers\Select.h"
#include "modules\generators\DecarpientierSwiss.h"
using namespace cnoise::generators;

int main() {
	static int img_size = 8192;
	int i = 0;
	Billow2D ptb(img_size, img_size, noise_t::SIMPLEX, 0.12232f, -1.232f, 5474, 0.0003f, 1.60f, 8, 0.70f);
	ptb.Generate();
	//ptb.SaveToPNG("dc_swiss_perlin.png");
	ptb.SaveToTER("decarpienter.ter");
}