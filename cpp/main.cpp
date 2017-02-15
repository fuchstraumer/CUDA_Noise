#include "stdafx.h"
#include "modules\Billow.h"

int main() {
	noise::module::Billow2D module(1024, 512);
	module.Generate();
}