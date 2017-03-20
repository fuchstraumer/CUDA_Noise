// Include modules
#define USING_CNOISE_NAMESPACES
#include "cnoise.h"
#include <ctime>

int main() {

	//std::clock_t timer;
	//int i = 0;
	//Billow3D test(4096, 4096, 13213123, 0.4f, 1.70f, 12, 0.90f);
	//Sphere test_projection(&test);
	//test_projection.SetSourceModule(&test);
	//test_projection.Build();
	//const auto pts = test.GetPoints();
	//test_projection.SaveToPNG("globe.png");

	constexpr int img_size_x = 8192;
	constexpr int img_size_y = 8192;
	constexpr float sea_level = 0.10f;
	constexpr float continent_freq = 1.5f;
	constexpr float continent_lacun = 2.10f;

	/*

	Group 1: Continent definition.

	*/

	/*Checkerboard check(img_size_x, img_size_y);
	check.Generate();
	check.SaveToPNG("checkerboard.png");
	Turbulence debug(img_size_x, img_size_y, noise_t::SIMPLEX, &check, 7, 331414, 6.0f, 1.50f);
	debug.Generate();
	debug.SaveToPNG("debug.png");*/

	using namespace cnoise;

	// Base continent module.
	FBM3D baseContinentDefinition_pe0(img_size_x, img_size_y, 123132, continent_freq, continent_lacun, 13, 0.70f);

	// Curve output so that very high values are near sea level, which defines the positions of the mountain ranges.
	Curve3D baseContinentDefinition_cu0(img_size_x, img_size_y);
	baseContinentDefinition_cu0.ConnectModule(&baseContinentDefinition_pe0);
	std::vector<cnoise::ControlPoint> base_cu0_pts = {
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
	FBM3D baseContinentDefinition_pe1(img_size_x, img_size_y, 4674, continent_freq * 4.35f, continent_lacun, 15, 0.75f);

	// This scales the output from the previous to be mostly near 1.0
	ScaleBias3D baseContinentDef_sb(img_size_x, img_size_y, 0.375f, 0.625f);
	baseContinentDef_sb.ConnectModule(&baseContinentDefinition_pe1);

	// This module carves out chunks from the curve module used to set ranges for continents, by selecting the min
	// between the carver chain and the base continent chain.
	Min3D baseContinentDef_min0(img_size_x, img_size_y, &baseContinentDef_sb, &baseContinentDefinition_cu0);


	Cache3D baseContinentDef_ca0(img_size_x, img_size_y, &baseContinentDef_min0);

	//Turbulence baseContinentDef_tu0(img_size_x, img_size_y, noise_t::PERLIN, &baseContinentDef_cl, 13, 1341324, 10.0f, continent_freq);
	//baseContinentDef_tu0.Generate();
	////cudaDeviceSynchronize();
	//baseContinentDef_tu0.SaveToPNG("turbulence.png");
	

	Sphere Test_Projection(&baseContinentDef_ca0);
	Test_Projection.Build();
	baseContinentDefinition_cu0.Generate();
	baseContinentDefinition_cu0.freeChildren();
	baseContinentDef_sb.Generate();
	baseContinentDef_sb.freeChildren();
	baseContinentDef_min0.Generate();
	baseContinentDef_min0.freeChildren();
	baseContinentDef_ca0.Generate();
	baseContinentDef_ca0.SaveToPNG("cache.png");
	return 0;
}