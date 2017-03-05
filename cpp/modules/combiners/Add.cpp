#include "Add.h"
#include "../cuda/combiners/add.cuh"
namespace cnoise {

	namespace combiners {

		Add::Add(int width, int height, float add_value, Module* source) : Module(width, height) {
			sourceModules.front() = source;
		}

		void Add::Generate(){
			// Make sure all modules in source modules container are generated. (and that there are no null module connections)
			for (const auto& module : sourceModules) {
				if (module == nullptr) {
					std::cerr << "Did you forget to attach a source module(s) to your add module?" << std::endl;
					throw("No source module(s) for Add module, abort");
				}
				if (!module->Generated) {
					module->Generate();
				}
			}
			
			AddLauncher(Output, sourceModules[0]->Output, sourceModules[1]->Output, dims.first, dims.second);
			Generated = true;
		}

		size_t Add::GetSourceModuleCount() const{
			return 2;
		}

	}
}

