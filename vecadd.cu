/* File:     vec_add.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

/* Kernel for vector addition */
__global__ void Vec_add(float x[], float y[], float z[], int n) {
  /* blockDim.x = threads_per_block                            */
  /* First block gets first threads_per_block components.      */
  /* Second block gets next threads_per_block components, etc. */
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  /* block_count*threads_per_block may be >= n */
  if (i < n) z[i] = x[i] + y[i];
}  /* Vec_add */

/* Host code */
int main() {
  //create some vectors, run the test for each one and report the output. This will be good C++ and CUDA practice.
  int n, i;
  float *h_x, *h_y, *h_z;
  float *d_x, *d_y, *d_z;
  int threads_per_block;
  int block_count;
  size_t size;

  n = 10000000; // Number of elements in vector
  size = n*sizeof(float); // Vector size

  /* Allocate input vectors in host memory */
  h_x = (float*) malloc(size);
  h_y = (float*) malloc(size);
  h_z = (float*) malloc(size);

  /* Initialize input vectors */
  for (i = 0; i < n; i++) {
    h_x[i] = i+1;
    h_y[i] = n-i;
  }

  /* Allocate vectors in device memory */
  // We use host pointers as a pointer to the on-device memory.
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);
  cudaMalloc(&d_z, size);

  /* Copy vectors from host memory to device memory */
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

  /* Define block size */
  threads_per_block = 256;

  /* Define grid size.  If we just computed n/threads_per_block */
  /* we might get fewer threads than vector components.  Using  */
  /* ceil(n/threads_per_block) guarantees at least one thread   */
  /* per vector component.  The following formula is a kludge   */
  /* since it appears that the CUDA ceil function doesn't work  */
  /* correctly.                                                 */
  block_count = (n + threads_per_block - 1)/threads_per_block; // just enough blocks that there is a thread per element

  /* Invoke kernel using block_count blocks, each of which  */
  /* contains threads_per_block threads                     */

  Vec_add<<<block_count, threads_per_block>>>(d_x, d_y, d_z, n);
  cudaThreadSynchronize();
  int numTests = 10000;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  for (int i =0; i<numTests; i++){
    Vec_add<<<block_count, threads_per_block>>>(d_x, d_y, d_z, n);
    /* Wait for the kernel to complete */
  }
  cudaThreadSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>( t2 - t1 ).count();

  cout << (duration/1000) << " ms" << endl;

  float numSeconds = (float) duration / 1e6;
  long long numFlops = long(n) * long(numTests);
  float flopsPerSecond = (float)numFlops / numSeconds;
  float MflopsPerSecond = flopsPerSecond / 1e6;
  cout << "MFLOP/s = " << MflopsPerSecond << endl;

  /* Copy result from device memory to host memory */
  /* h_z contains the result in host memory        */
  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

  float expectedSum = n + 1;
  printf("Testing....");
  for (i = 0; i < n; i++){
    if(h_z[i] != expectedSum){
      printf("Failure at %i ", i);
    }
  }
  printf("\n");

  /* Free device memory */
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  /* Free host memory */
  free(h_x);
  free(h_y);
  free(h_z);

  return 0;
}  /* main */
