#include "DecarpientierSwiss.h"
#include "../cuda/generators/decarpientier_swiss.cuh"

cnoise::generators::DecarpientierSwiss::DecarpientierSwiss(int width, int height, noise_t noise_type, float x, float y, int seed, float freq, float lacun,
	int octaves, float persist) : Module(width, height), Attributes(seed, freq, lacun, octaves, persist), Origin(x, y), NoiseType(noise_type){}

size_t cnoise::generators::DecarpientierSwiss::GetSourceModuleCount() const{
	return 0;
}

void cnoise::generators::DecarpientierSwiss::Generate(){
	DecarpientierSwissLauncher(Output, dims.first, dims.second, NoiseType, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
	Generated = true;
}

cnoise::generators::DecarpientierSwiss3D::DecarpientierSwiss3D(int width, int height, int seed, float freq, float lacun, int octaves, float persist): Attributes(seed, freq, lacun, octaves, persist), Module3D(nullptr, width, height) {}

size_t cnoise::generators::DecarpientierSwiss3D::GetSourceModuleCount() const{
	return 0;
}

void cnoise::generators::DecarpientierSwiss3D::Generate(){
	DecarpientierSwissLauncher3D(Points, dimensions.x, dimensions.y, Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
	Generated = true;
}
