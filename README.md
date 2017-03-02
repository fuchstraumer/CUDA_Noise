# CUDA_Noise
  CUDA-accelerated module-based noise, mainly accessed through a C++ wrapper to kernel launches and the like. WIP. In need of serious optimization when it comes to the CUDA code. Wiki is WIP too, this is inspired by libnoise and I'd like to match their documentation as best as I can. Should be release-ready by the end of next month: feel free to play with it until then, but its not really documented at all yet.
  
  
  Unfortunately, I don't have any plans to port this to OpenCL or HIP at this time. It is an option I'd like to explore, but the majority of the modular nature of this library requires using things like texture memory and texture objects - something that HIP does not support, and something I don't know how to use in OpenCL.
  
  
  Due to HIPs current lack of many key features, I'll need to to what will amount to an OpenCL rewrite when I have time. School is picking back up - so this probably won't be soon. Apologies to non-Nvidia GPU-owning individuals.
 
 ---
## Credit where credit is due

  This library stands on the work of many other libraries before it - I have attempted to provide attribution in several locations within my code, but I'd like to also take the time to explicitly mention the key repositories and projects that have helped me with this project thus far:
  
  
1. First, libnoise provided the inspiration for the module system.
 
  * http://libnoise.sourceforge.net/
 
  * The Perlin CUDA code is directly taken from Libnoise - along with the logic, naming scheme, and layout of much of the rest of my code. Libnoise may be dated, but its still a remarkable project and I owe it a tremendous amount.
 
2. Second, FastNoise and FastNoiseSIMD have provided inspiration for various optimized versions of the noise generators in particular. @Auburns has done some remarkable work. The contraints brought about by SIMD programming have many parallels in GPU programming, and as such his code continues to be something I play with as I seek to improve my own code
  
  * https://github.com/Auburns 
 
3. Third, Stefan Gustavson's simplex noise work and code is practically a reference implementation of simplex noise at this point - and the wealth of information he has provided and continus to provided has been of tremendous aid.
    
  * http://weber.itn.liu.se/~stegu/
  
  
4. Fourth, AccidentalNoise has provided WONDERFUL hashing methods and algorithms that should let me ditch having to pass LUTs, be they in constant memory or encapsulated in texture objects, to CUDA. While still keeping the ability to "seed" the function! (missing in many GPU-friendly/shadercode implementations of noise generation)
  
   * http://accidentalnoise.sourceforge.net/
    
 I could not have completed any of this project without the above resources and code. A tremendous thanks to them, and doubly so for being open source!
 
---

### Example output:
  
  * Select module w/ falloff:
  
  * FBM module:
  
  * Ridged-multi module:
  
  * Example of increasing octaves from 1 - 10
  
--- 
  
  
### In-Use and setup:
