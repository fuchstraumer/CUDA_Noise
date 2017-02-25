#pragma once
#ifndef CURVE_H
#define CURVE_H
#include "common\CommonInclude.h"
#include "common\CUDA_Include.h"
#include "..\Base.h"
/*

	Modifier module - Curve

	An expandable class that curves data. Has an internal vector of control points
	that can be expanded and is used to define the curve. This vector is passed
	to the CUDA kernel.

	NOTE: Unlike other modules, this module MUST be setup before using, if a vector
	of control points is not supplied in the constructor. This is due ot how we have
	to set everything in CUDA up.

*/

namespace noise {

	namespace module {

		struct ControlPoint {

			ControlPoint(float in, float out) : InputVal(in), OutputVal(out) {}

			// Input value, or "x"
			float InputVal;

			// Output value, or "y"
			float OutputVal;

		};


		class Curve : public Module {
		public:

			// Doesn't add any control points. Empty constructor.
			Curve(int width, int height);

			// Adds control points from given vector and makes sure kernel is good to go ASAP
			Curve(int width, int height, const std::vector<ControlPoint>& init_points);

			// Destructor.
			~Curve();

			// Prepares CUDA data before launching the kernel.
			void PrepareData();

			// Adds a control point
			void AddControlPoint(float input_val, float output_val);

			// Get control points (non-mutable)
			std::vector<ControlPoint> GetControlPoints() const;

			// Clear control points
			void ClearControlPoints();

			// Whether or not this object is ready to be launched.
			bool Ready;

			// Connect a source module to this object.
			virtual void ConnectModule(Module& other) override;

			// Generate data by launching the CUDA kernel
			virtual void Generate() override;

		protected:

			// Set if points are added after we already prepared the CUDA array, in which case we
			// need to rebuild all the CUDA data.
			bool update;

			// Rebuild CUDA data
			void rebuildCUDA_Data();

			// Control points.
			std::vector<ControlPoint> controlPoints;

			// CUDA Array holds actual data: CUDA texture object is itnerface to this
			cudaArray* cpArray;

			// CUDA texture is passed to the kernel and contains the control points.
			// This will be a two-channel texture of 32-bit floats. Dimensions found
			// from size of control point vector, but will always have a y dim of 2.
			cudaTextureObject_t cpTex;
		};

	}

}
#endif // !CURVE_H
