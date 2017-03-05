// Include modules
#define USING_CNOISE_NAMESPACES
#include "modules\Modules.h"

inline void PrintMemInfo() {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cerr << " Total allocated: " << total - free << std::endl;
}

int main() {

	constexpr int img_size_x = 8192;
	constexpr int img_size_y = 8192;
	constexpr float sea_level = 0.10f;
	constexpr float continent_freq = 0.00025f / 8;
	constexpr float continent_lacun = 2.10f;

	/*
		
		Group 1: Continent definition.
	
	*/

	// Base continent module.
	FBM2D baseContinentDefinition_pe0(img_size_x, img_size_y, noise_t::PERLIN, 0.0f, 0.0f, 123132, continent_freq, continent_lacun, 10, 0.60f);

	// Curve output so that very high values are near sea level, which defines the positions of the mountain ranges.
	Curve baseContinentDefinition_cu0(img_size_x, img_size_y);
	baseContinentDefinition_cu0.ConnectModule(&baseContinentDefinition_pe0);
	std::vector<ControlPoint> base_cu0_pts = {
		ControlPoint(-2.00f + sea_level, -1.625f + sea_level),
		ControlPoint(-1.00f + sea_level, -1.375f + sea_level),
		ControlPoint(0.000f + sea_level, -0.375f + sea_level),
		ControlPoint(0.0625f + sea_level, 0.125f + sea_level),
		ControlPoint(0.1250f + sea_level, 0.250f + sea_level),
		ControlPoint(0.2500f + sea_level, 1.000f + sea_level),
		ControlPoint(0.5000f + sea_level, 0.250f + sea_level),
		ControlPoint(0.7500f + sea_level, 0.250f + sea_level),
		ControlPoint(1.0000f + sea_level, 0.500f + sea_level),
		ControlPoint(2.0000f + sea_level, 0.500f + sea_level),
	};
	baseContinentDefinition_cu0.SetControlPoints(base_cu0_pts);

	// This is used to carve out chunks from the mountain ranges, so that they don't become entirely impassable.
	FBM2D baseContinentDefinition_pe1(img_size_x, img_size_y, noise_t::PERLIN, 0.0f, 0.0f, 2234121, continent_freq * 4.35f, continent_lacun, 11, 0.70f);

	// This scales the output from the previous to be mostly near 1.0
	ScaleBias baseContinentDef_sb(img_size_x, img_size_y, 0.375f, 0.625f);
	baseContinentDef_sb.ConnectModule(&baseContinentDefinition_pe1);

	// This module carves out chunks from the curve module used to set ranges for continents, by selecting the min
	// between the carver chain and the base continent chain.
	Min baseContinentDef_min0(img_size_x, img_size_y, &baseContinentDef_sb, &baseContinentDefinition_cu0);

	// Finally, clamp module clamps min value output to be between -1.0 and 1.0.
	Clamp baseContinentDef_cl(img_size_x, img_size_y, -1.0, 1.0f, &baseContinentDef_min0);
	baseContinentDef_cl.Generate();
	baseContinentDef_cl.SaveToPNG("terrain.png");
	cudaDeviceSynchronize();

	Turbulence baseContinentDef_tu0(img_size_x, img_size_y, noise_t::PERLIN, &baseContinentDef_cl, 13, 1341324, 1.0f / continent_freq, continent_freq * 15.0f);
	baseContinentDef_tu0.Generate();
	cudaDeviceSynchronize();
	baseContinentDef_tu0.SaveToPNG("turbulence.png");

}