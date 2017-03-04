#pragma once
#ifndef TURBULENCE_H
#define TURBULENCE_H
#include "../Base.h"

namespace cnoise {

	namespace modifiers {

		constexpr int DEFAULT_TURBULENCE_ROUGHNESS = 3;
		constexpr int DEFAULT_TURBULENCE_SEED = 0;
		constexpr float DEFAULT_TURBULENCE_STRENGTH = 3.0f;

		class Turbulence : public Module {
		public:

			Turbulence(int width, int height, noise_t noise_type, Module* prev = nullptr, int roughness = DEFAULT_TURBULENCE_ROUGHNESS, int seed = DEFAULT_TURBULENCE_SEED, float strength = DEFAULT_TURBULENCE_STRENGTH);

			virtual size_t GetSourceModuleCount() const override;

			virtual void Generate() override;

			void SetNoiseType(noise_t _type);

			noise_t GetNoiseType() const;

			void SetStrength(float _strength);

			float GetStrength() const;

			void SetSeed(int _seed);
			
			int GetSeed() const;

			void SetRoughness(int _rough);

			int GetRoughness() const;

		private:

			float strength;

			int seed;

			int roughness;

			noise_t noiseType;

		};
	
	}

}

#endif // !TURBULENCE_H
