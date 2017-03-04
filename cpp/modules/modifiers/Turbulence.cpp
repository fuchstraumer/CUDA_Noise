#include "Turbulence.h"
#include "../cuda/modifiers/turbulence.cuh"
namespace cnoise {
	
	namespace modifiers {

		Turbulence::Turbulence(int width, int height, noise_t noise_type, Module* prev, int _roughness, int _seed, float _strength) : Module(width, height), roughness(_roughness), seed(_seed), strength(_strength) {
			ConnectModule(prev);
		}

		size_t Turbulence::GetSourceModuleCount() const{
			return 1;
		}

		void Turbulence::Generate(){
			if (sourceModules.front() == nullptr) {
				std::cerr << "Did you forget to set a source module for your turbulence module?" << std::endl;
				throw("No source module for turbulence module set before attempting to generate results.");
			}
			if (!sourceModules.front()->Generated) {
				sourceModules.front()->Generate();
			}
			TurbulenceLauncher(output, sourceModules.front()->output, dims.first, dims.second, noiseType, roughness, seed, strength);
			Generated = true;
		}

		void Turbulence::SetNoiseType(noise_t _type){
			noiseType = _type;
		}

		noise_t Turbulence::GetNoiseType() const{
			return noiseType;
		}

		void Turbulence::SetStrength(float _strength){
			strength = _strength;
		}

		float Turbulence::GetStrength() const{
			return strength;
		}

		void Turbulence::SetSeed(int _seed){
			seed = _seed;
		}

		int Turbulence::GetSeed() const{
			return seed;
		}

		void Turbulence::SetRoughness(int _rough) {
			roughness = _rough;
		}

		int Turbulence::GetRoughness() const {
			return roughness;
		}

	}

}