#pragma once
#ifndef MODULES_H
#define MODULES_H

// Include base so its easier to attach modules and such
#include "Base.h"

// Include modules.
#include "combiners\Combiners.h"
#include "generators\Generators.h"
#include "modifiers\Modifiers.h"
#include "utility\Utility.h"

// Simple macro to shortcut away the namespaces.
#ifdef USING_CNOISE_NAMESPACES
using namespace cnoise::combiners;
using namespace cnoise::generators;
using namespace cnoise::modifiers;
using namespace cnoise::utility;
#endif // USING_CNOISE_NAMESPACES


#endif // !MODULES_H
