"""
Multiply a 30x30 matrix by 2. This is a very small operation and inherently not suited to GPU, however it provides a good demonstration.
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time

# Generate some random data:
a = numpy.random.randn(30,30)
# Convert to f32 as Nvidia devices don't support double precision
a = a.astype(numpy.float32)

# Allocate memory on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
# Do a memcpy, data is now on the GPU global memory
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*30;
    a[idx] *= 2;
  }
  """)

func = mod.get_function("doublify")
a_doubled = numpy.empty_like(a)

t0=time.time()
func(a_gpu, block=(30,30,1)) # Async call
cuda.memcpy_dtoh(a_doubled, a_gpu)
tf=time.time()

total_time = tf-t0
gflops = a.size/total_time * 1e-9
print('GFLOP/s: ', gflops)
