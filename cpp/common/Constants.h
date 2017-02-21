#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H

/*
	
	CONSTANTS_H

	Used to store/set various program-wide constants.

*/

// Noise types.

namespace noise {
	namespace module {

		// Types of base noise available 
		enum class NoiseType {
			PERLIN,
			SIMPLEX,
		};

	}
}

#endif // !CONSTANTS_H
