#pragma once
#ifndef BASE_H
#define BASE_H
#include "common\CommonInclude.h"
#include "common\CUDA_Include.h"
/*
	
	Defines a base module class.

	Each module inherits from this, so that we can
	link modules together safely.

	This mainly involves checking for compatible parameters
	between linked modules, and chaining together generate
	commands to create the final object.

*/
namespace noise {

	namespace module {

		class Module {
			// Delete copy ctor and operator
			Module(const Module& other) = delete;
			Module& operator=(const Module& other) = delete;
			// Implement move ctor and operator once this class is more fully implemented. (how to move CUDA data?)
		public:

			// Each module must have a width and height, as this specifies the size of 
			// the surface object a module will write to, and must match the dimensions
			// of the texture object the surface will read from.
			Module(int width, int height);

			// Destructor calls functions to clear CUDA objects/data
			~Module();

			// Connects this module to another source module
			virtual void ConnectModule(Module& other);

			// Generates data and stores it in this object
			virtual void Generate() = 0;

			// Returns Generated data.
			virtual cudaSurfaceObject_t GetData() const;

			// Gets reference to module at given index in this modules "sourceModules" container
			virtual std::shared_ptr<Module> GetModule(size_t idx) const;

			// Get number of source modules connected to this object.
			virtual int GetSourceModuleCount() const = 0;

			// Get texture data from GPU and return it as a vector of floating point values.
			virtual std::vector<float> GetGPUData() const;

			// Get texture from GPU and return it as a normalized (0.0 - 1.0) vector floating point values
			virtual std::vector<float> GetGPUDataNormalized() const;

			// Save current module to an image with name "name"
			virtual void SaveToPNG(const char* name);

			// Tells us whether or not this module has already Generated data.
			bool Generated;

			// Each module will write values into this
			cudaSurfaceObject_t output;

		protected:

			// Dimensions of textures.
			std::pair<int, int> dims;

			// underlying CUDA arrays that will hold our data.
			cudaArray *surfArray;

			

			// Modules that precede this module, with the back 
			// of the vector being the module immediately before 
			// this one, and the front of the vector being the initial
			// module.
			std::vector<std::shared_ptr<Module>> sourceModules;
		};

		// Config struct for noise generators.
		struct noiseCfg {

			// Seed for the noise generator
			int Seed;
			// Frequency of the noise
			float Frequency;
			// Lacunarity controls amplitude of the noise, effectively
			float Lacunarity;
			// Controls how many octaves to use during octaved noise generation
			int Octaves;
			// Persistence controls how the amplitude of successive octaves decreases.
			float Persistence;
			
			noiseCfg(int s, float f, float l, int o, float p) : Seed(s), Frequency(f), Lacunarity(l), Octaves(o), Persistence(p) {}

			noiseCfg() = default;
			~noiseCfg() = default;

		};

	}
}


#endif // !BASE_H
