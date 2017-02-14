#pragma once
#ifndef CONTEXT_H
#define CONTEXT_H
#include "stdafx.h"
#include "cuda_stdafx.cuh"
#include "Shader.h"
/*
	
	Class - Context

	Used to initialize and activate an OpenGL rendering
	context. 

	Tuned from my source code at:
	https://github.com/fuchstraumer/DiamondDogs/blob/master/DiamondDogs/engine/Context.h
	https://github.com/fuchstraumer/DiamondDogs/blob/master/DiamondDogs/engine/Context.cpp
	to work with this application and in 2D

	Using example code at:
	https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st/blob/master/

	For creating CUDA interface to modern OpenGL: shaders are what I know, and immediate-mode
	stuff gets really hard with larger programs and rendering objects!

	GLFW Compatability/interop demonstrated at:
	https://gist.github.com/allanmac/4ff11985c3562830989f

*/



class Context {
	// Removing ability to move or copy this object,
	// OpenGL really doesn't like if I do that

	// Copy operator/assigment
	Context& operator=(const Context& other) = delete;
	// Copy ctor
	Context(const Context& other) = delete;
	// Move operator/assignment
	Context& operator=(Context&& other) = delete;
	// Move ctor
	Context(Context&& other) = delete;

public:

	// CTor, width + height set dimensions. Inits GL.
	Context(const GLsizei& width, const GLsizei& height);

	// Init CUDA
	void CUDA_Initialize();

	// Activates this context and runs the rendering loop until the escape key is pressed.
	void Use();

	// Runs CUDA simulation for a step.
	void CUDA_Step(float dt);

	// GLFW function for processing mouse button presses
	static void MouseButtonCallback(GLFWwindow* window, int button, int code, int mods);

	// GLFW function for processing mouse movement
	static void MouseMovementCallback(GLFWwindow* window, double x_offset, double y_offset);

	// GLFW function for processing keyboard actions
	static void KeyCallback(GLFWwindow* window, int key, int code, int actions, int mods);

	// Used to avoid odd/weird/uneven update times
	GLfloat LastFrame, DeltaTime;

	// We use a simple VBO/VAO defining a quad as our main renderable item that we draw to.
	GLuint VBO, VAO;

	// Shader used for this context
	ShaderProgram MainShader;

	// We need two texture handles: we swap them back and forth.
	// The latest updated texture gets rendered, one needing update gets sent to CUDA, 
	// next frame we swap these: hence, "Ping-Pong" method in example above.
	GLuint Texture;

	// These arrays are used as CUDA's reference to the textures
	cudaArray* CUDA_Array;

	// We use CUGraphicsResource to register a resource that will be rendered with CUDA
	// This is like a resource that is shared between CUDA and OpenGL
	cudaGraphicsResource_t CUDA_Texture;

	// Object passed to the kernels, surface for writing
	cudaSurfaceObject_t CUDA_Surface;

	GLFWwindow* Window;

};

#endif // !CONTEXT_H
