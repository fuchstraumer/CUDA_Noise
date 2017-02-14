#version 330 core

layout(location = 0) in vec3 pos;

out vec2 uv;

void main() {
	gl_Position = vec4(pos, 1.0f);
	uv = vec2(pos.xy) * 0.50f + 0.50f;
}