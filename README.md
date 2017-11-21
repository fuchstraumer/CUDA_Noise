# CUDA_Noise
  CUDA-accelerated module-based noise, mainly accessed through a C++ wrapper to kernel launches and the like. Undocumented, mostly untested, and has a number of large errors - sorry. When it does work, though, it's real bloody fast!

 ---
## Other projects of use/note

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
  
  
### In-Use and setup:
