#pragma once
#ifndef QUAD_H
#define QUAD_H
#include "stdafx.h"

/*
	QUAD_H

	Defines a simple quadrilateral that we use for rendering 
	to, with our FBO. This quad acts as the "surface" we write
	our texture onto.

	We can combine multiple instances of these, or chain them
	together, to get really complex rendering effects. Especially
	with post-processing stuff for whole-scene rendering.

	Not sure how that works with CUDA though.
*/


static constexpr GLfloat quad_vertices[12]{
	-0.5f,-0.5f, 0.0f,
	 0.5f,-0.5f, 0.0f,
	-0.5f, 0.5f, 0.0f,
	 0.5f, 0.5f, 0.0f,
};


#endif // !QUAD_H
