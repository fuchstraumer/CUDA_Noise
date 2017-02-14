#include "stdafx.h"
#include "context.h"
#include "..\cuda_assert.h"


// Used to find which keys are active/toggled
static bool keys[1024];
static bool keysRelease[1024];

// Previous mouse position
static GLfloat lastX = static_cast<GLfloat>(SCR_WIDTH) / 2.0f;
static GLfloat lastY = static_cast<GLfloat>(SCR_HEIGHT) / 2.0f;
// Used to set proper 
static bool mouseInit = true;

Context::Context(const GLsizei & width, const GLsizei & height) : LastFrame(0.0f), DeltaTime(0.0f) {

	// Init glfw
	glfwInit();

	// Set OpenGL version to use: 4.3 Core Profile
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Disable window resizing.
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create window instance.
	Window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), "CUDA_Context", nullptr, nullptr);
	
	// Make sure GLFW window creation worked.
	if (Window == nullptr) {
		std::cerr << "GLFW Window creation failed." << std::endl;
		throw("CONTEXT::CONTEXT::L28: Creation of a GLFW window failed.");
	}

	// Set generated window as current rendering context.
	glfwMakeContextCurrent(Window);

	// Set callbacks
	glfwSetCursorPosCallback(Window, MouseMovementCallback);
	glfwSetMouseButtonCallback(Window, MouseButtonCallback);
	glfwSetKeyCallback(Window, KeyCallback);

	// Initialize the actual OpenGL context
	GLuint init = glewInit();

	// Make sure we initialized successfully 
	if (init != GLEW_OK) {
		std::cerr << "OpenGL/GLEW failed to initialize: is OpenGL installed on the current sytem? Is there an available context?" << std::endl;
		// Teminate GLFW context since GLEW failed to init
		glfwTerminate();
		throw("CONTEXT::CONTEXT::L32: GLEW Initialization failed!");
	}

	// Check to make sure version 4.3 is support
	if (!GLEW_VERSION_4_3) {
		std::cerr << "OpenGL 4.3 is not supported on the current system, exiting." << std::endl;
		// Terminate GLFW context
		glfwTerminate();
		throw("CONTEXT::CONTEXT::L40: Current platform does not support OpenGL 4.3.");
	}

	// Disable depth testing, we're rendering in 2D ehre
	glDisable(GL_DEPTH_TEST);

	// Set viewport.
	glViewport(0, 0, width, height);

	// Setup shader
	MainShader.Init();

	// Create sub-programs.
	Shader vert("./shaders/vertex.glsl", VERTEX_SHADER); 
	Shader frag("./shaders/fragment.glsl", FRAGMENT_SHADER);
	
	// Attach shaders
	MainShader.AttachShader(vert);
	MainShader.AttachShader(frag);

	// Compile and link the completed program
	MainShader.CompleteProgram();

	// Activate our shader.
	MainShader.Use();

	// Some screens use DPI scaling and such, so before setting up our renderbuffers and
	// textures that we render to, lets make sure that "width" and "height" are set
	// to the right value (note: this doesn't affect the glViewport)
	int fwidth = static_cast<int>(width); 
	int fheight = static_cast<int>(height);
	glfwGetFramebufferSize(Window, &fwidth, &fheight);

	// Generate texture.
	glGenTextures(1, &Texture);

	// Bind texture.
	glBindTexture(GL_TEXTURE_2D, Texture);

	// Tell OpenGL to allocate for an empty texture with format matching our renderbuffer
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, width, height);
	//glTexImage2D(GL_TEXTURE_2D, 1, GL_RGBA8UI, fwidth, fheight, 0, GL_RGBA8UI, GL_UNSIGNED_BYTE, nullptr);

	// Simple filtering and texture clamping.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


	// Data used for the quad we render to
	static const GLfloat quad_vertices[18]{
	   -1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	   -1.0f,  1.0f, 0.0f,
	   -1.0f,  1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f,  1.0f, 0.0f,
	};

	// Gen VAO
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	// Setup the VBO and load it with the data from quad vertices.
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

	// Set our single VAO attribute, position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (GLvoid*)0);

	// Unbind our buffer and VAO
	// Note: turns out this actually isn't needed.
	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindVertexArray(0);
	//glBindTexture(GL_TEXTURE_2D, 0);
	CUDA_Initialize();
}

void Context::CUDA_Initialize(){
	// Attempt to register OpenGL resources with CUDA
	cudaError_t err = cudaSuccess;
	err = cudaGraphicsGLRegisterImage(&CUDA_Texture, Texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	cudaAssert(err);

	// Map the resource
	err = cudaGraphicsMapResources(1, &CUDA_Texture, 0);
	cudaAssert(err);

	// Bind texture to array
	err = cudaGraphicsSubResourceGetMappedArray(&CUDA_Array, CUDA_Texture, 0, 0);
	cudaAssert(err);

	// Unmap the resource, now that we've bound it to the array
	err = cudaGraphicsUnmapResources(1, &CUDA_Texture, 0);
	cudaAssert(err);

	// Setup resource description object and bind the CUDA Array to it
	struct cudaResourceDesc rsrc;

	// Zero-initialize the resource description
	memset(&rsrc, 0, sizeof(rsrc));
	rsrc.resType = cudaResourceTypeArray;
	rsrc.res.array.array = CUDA_Array;

	// Now set up the surface object, which is what the kernels will write to.
	CUDA_Surface = 0;
	// Second argument is resource description, third argument can be a cudaTextureDesc
	// describing how to interpret the texture along with any extra parameters that may come up
	// (filtering, border color, mipmaps, sRGB conversion, etc)
	err = cudaCreateSurfaceObject(&CUDA_Surface, &rsrc);
	cudaAssert(err);
}

void Context::Use(){

	// Set active texture unit
	glActiveTexture(GL_TEXTURE0);

	// Make sure shader is active
	MainShader.Use();

	// While glfwWindowShouldClose flag is not set, run this rendering loop
	while (!glfwWindowShouldClose(Window)) {

		// get time elapsed between current and last frame, and pass it 
		// to the function that calls the CUDA kernel
		GLfloat current_frame = static_cast<GLfloat>(glfwGetTime());
		DeltaTime = current_frame - LastFrame;
		LastFrame = current_frame;

		// Call CUDA kernel with elapsed time.
		CUDA_Step(DeltaTime);

		// Clear color buffer to black
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Poll events, check for escape key etc
		glfwPollEvents();

		// Draw elements to screen.
		glDrawArrays(GL_TRIANGLES, 0, 6);

		// Swap buffers used for double-buffered rendering.
		glfwSwapBuffers(Window);
	}

	// Terminate GLFW
	glfwTerminate();

	// Clean up texture.
	glDeleteTextures(1, &Texture);

	// Unregister resources
	cudaGraphicsUnregisterResource(CUDA_Texture);

}

void Context::CUDA_Step(float dt){
	// Launch kernel.
	
}


void Context::MouseButtonCallback(GLFWwindow * window, int button, int code, int mods){
	
}

void Context::MouseMovementCallback(GLFWwindow * window, double x_offset, double y_offset){
	if (mouseInit) {
		lastX = static_cast<GLfloat>(x_offset);
		lastY = static_cast<GLfloat>(y_offset);
		mouseInit = false;
	}
	// Keep track of mouse position
	// Object picking?
	GLfloat xoffset = static_cast<GLfloat>(x_offset) - lastX;
	GLfloat yoffset = static_cast<GLfloat>(y_offset) - lastY;

	lastX = static_cast<GLfloat>(x_offset);
	lastY = static_cast<GLfloat>(y_offset);
}

void Context::KeyCallback(GLFWwindow * window, int key, int code, int action, int mods){

	// Tells window to close and exits rendering loop.
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	// Allows the mouse to enter/leave
	if (key == GLFW_KEY_LEFT_ALT && action == GLFW_PRESS) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
	if (key == GLFW_KEY_LEFT_ALT && action == GLFW_RELEASE) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	
}
