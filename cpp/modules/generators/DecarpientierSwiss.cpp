#include "DecarpientierSwiss.h"
#include "../cuda/generators/decarpientier_swiss.cuh"

noise::module::DecarpientierSwiss::DecarpientierSwiss(int width, int height, noise_t noise_type, float x, float y, int seed, float freq, float lacun,
	int octaves, float persist) : Module(width, height), Attributes(seed, freq, lacun, octaves, persist), Origin(x, y), NoiseType(noise_type){}

int noise::module::DecarpientierSwiss::GetSourceModuleCount() const{
	return 0;
}

void noise::module::DecarpientierSwiss::Generate(){
	DecarpientierSwissLauncher(output, dims.first, dims.second, NoiseType, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
}
