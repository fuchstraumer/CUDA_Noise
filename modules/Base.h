#pragma once
#ifndef BASE_H
#define BASE_H
#include "cuda_stdafx.cuh"
#include <memory>
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

			// Returns generated data.
			virtual cudaSurfaceObject_t* GetData() const = 0;

			// Gets reference to module at given index in this modules "sourceModules" container
			virtual std::shared_ptr<Module> GetModule(size_t idx) const;

			// Get number of source modules connected to this object.
			virtual int GetSourceModuleCount() const = 0;

		protected:

			// Dimensions of textures.
			glm::ivec2 dims;

			// Each module will write values into this
			cudaSurfaceObject_t output;

			// Each module can read values from this.
			cudaTextureObject_t input;

			// underlying CUDA arrays that will hold our data.
			cudaArray *surfArray, *texArray;

			// Tells us whether or not this module has already generated data.
			bool generated;

			// Modules that precede this module, with the back 
			// of the vector being the module immediately before 
			// this one, and the front of the vector being the initial
			// module.
			std::vector<std::shared_ptr<Module>> sourceModules;
		};

		// Config struct for noise generators.
		struct noiseCfg {

			// Frequency of the noise
			float Frequency;
			// Lacunarity controls amplitude of the noise, effectively
			float Lacunarity;
			// Persistence controls how the amplitude of successive octaves decreases.
			float Persistence;
			// Controls how many octaves to use during octaved noise generation
			int Octaves;
			// Seed for the noise generator
			int Seed;
			// Sets limit on octaves.
			int MaxOctaves;

			noiseCfg(float f, float l, float p, int o, int s, int max_o) : Frequency(f), Lacunarity(l), 
				Persistence(p), Octaves(o), Seed(s), MaxOctaves(max_o) {}

			noiseCfg() = default;
			~noiseCfg() = default;

		};

	}
}


#endif // !BASE_H
