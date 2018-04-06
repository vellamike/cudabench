import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time

# Let's make some random data:

a = numpy.random.randn(30,30)

a = a.astype(numpy.float32) # Convert to f32 as Nvidia devices don't support double precision


a_gpu = cuda.mem_alloc(a.nbytes) # Allocated memory on the GPU

cuda.memcpy_htod(a_gpu, a) # Do a memcpy, data is now on the GPU global memory.

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
func(a_gpu, block=(32,32,1)) # blocking call?
cuda.memcpy_dtoh(a_doubled, a_gpu)
tf=time.time()

total_time = tf-t0
gflops = a.size/total_time * 1e-9
print('GFLOP/s: ', gflops)


#print(a)
#print(a_doubled)
