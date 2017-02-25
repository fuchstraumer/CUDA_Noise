#include "Select.h"
#include "..\cuda\modifiers\select.cuh"
namespace noise {

	namespace module {

		Select::Select(int width, int height, float low_value, float high_value, float _falloff, std::shared_ptr<Module> selector, std::shared_ptr<Module> subject0, std::shared_ptr<Module> subject1) : Module(width, height), lowThreshold(low_value), highThreshold(high_value), falloff(_falloff)  {
			sourceModules.push_back(selector);
			sourceModules.push_back(subject0);
			sourceModules.push_back(subject1);
		}

		Select::~Select() {
			sourceModules[0]->~Module();
			sourceModules[1]->~Module();
			sourceModules[2]->~Module();
			// Delete cuda objects
			cudaDestroySurfaceObject(output);
			cudaFreeArray(surfArray);
		}

		void Select::SetSubject(size_t idx, std::shared_ptr<Module> subject){
			if (idx > 2 || idx < 1) {
				std::cerr << "Index supplied to SetSubject method of a Select module must be 1 or 2 - First subject, or second subject." << std::endl;
				throw("Invalid index supplied");
			}
		}

		void Select::SetSelector(std::shared_ptr<Module> selector){
			sourceModules[0] = selector;
		}

		void Select::Generate() {
			if (!sourceModules[0]->Generated) {
				sourceModules[0]->Generate();
			}
			if (!sourceModules[1]->Generated) {
				sourceModules[1]->Generate();
			}
			if (!sourceModules[2]->Generated) {
				sourceModules[1]->Generate();
			}

			if (sourceModules[0] == nullptr || sourceModules[1] == nullptr || sourceModules[2] == nullptr) {
				return;
			}

			SelectLauncher(output, sourceModules[0]->output, sourceModules[1]->output, sourceModules[2]->output, dims.first, dims.second, highThreshold, lowThreshold, falloff);
			Generated = true;
		}

		void Select::SetHighThreshold(float _high){
			highThreshold = _high;
		}

		void Select::SetLowThreshold(float _low)
		{
		}

		float Select::GetHighTreshold() const{
			return highThreshold;
		}

		float Select::GetLowThreshold() const{
			return lowThreshold;
		}

		void Select::SetFalloff(float _falloff){
			falloff = _falloff;
		}

		float Select::GetFalloff() const{
			return falloff;
		}

		int Select::GetSourceModuleCount() const {
			return 3;
		}

	}

}
