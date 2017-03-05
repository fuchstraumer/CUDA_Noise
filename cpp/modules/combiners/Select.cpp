#include "Select.h"
#include "..\cuda\combiners\select.cuh"
namespace cnoise {

	namespace combiners {

		Select::Select(int width, int height, float low_value, float high_value, float _falloff, Module* selector, Module* subject0, Module* subject1) : Module(width, height), lowThreshold(low_value), highThreshold(high_value), falloff(_falloff)  {
			sourceModules.resize(3);
			sourceModules[0] = selector;
			sourceModules[1] = subject0;
			sourceModules[2] = subject1;
		}

		void Select::SetSubject(size_t idx, Module* subject){
			if (idx > 2 || idx < 1) {
				std::cerr << "Index supplied to SetSubject method of a Select module must be 1 or 2 - First subject, or second subject." << std::endl;
				throw("Invalid index supplied");
			}
			else {
				sourceModules[idx] = subject;
			}
		}

		void Select::SetSelector(Module* selector){
			sourceModules[0] = selector;
		}

		void Select::Generate() {

			// Make sure modules are connected
			if (sourceModules[0] == nullptr || sourceModules[1] == nullptr || sourceModules[2] == nullptr) {
				throw;
			}

			// Make sure all source modules are generated.
			for (const auto& module : sourceModules) {
				if (!module->Generated) {
					module->Generate();
				}
			}

			SelectLauncher(Output, sourceModules[0]->Output, sourceModules[1]->Output, sourceModules[2]->Output, dims.first, dims.second, highThreshold, lowThreshold, falloff);
			Generated = true;
		}

		void Select::SetHighThreshold(float _high){
			highThreshold = _high;
		}

		void Select::SetLowThreshold(float _low){
			lowThreshold = _low;
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

		size_t Select::GetSourceModuleCount() const {
			return 3;
		}

	}

}
