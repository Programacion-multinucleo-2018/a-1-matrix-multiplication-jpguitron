#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>

using namespace std;

#define N 1000

//Fill the matrix with natural numbers starting with 0 (row major order)
void fillMatrix(long * matrix)
{
  int i;
  int size = N*N;
  for(i = 0; i < size; i++)
  {
      matrix[i] = i;
  }
}

//Print the matrix
void printMatrix(long * m_r)
{
  int size = N*N;
  int x;
  for(x = 0; x < size; x++)
  {

      if(x%N==0)
      {
        printf("\n");
      }
      printf("%ld ", m_r[x]);
  }
}
//Compare if two matrices are the same
bool checkResult(long * m_host, long * m_gpu)
{
  int size = N*N;
  for(int x = 0; x<size;x++)
  {
    if(m_host[x]!=m_gpu[x])
      return false;
  }
  return true;
}
//multiplication of matrices in cpu
void mulMatrix(long * m_r, long * m1, long * m2)
{
  int x;
  int y;
  int z;
  for(y=0;y<N;y++)
  {
    for(z = 0; z < N; z++)
    {
      for(x = 0; x < N; x++)
      {
          m_r[y*N+z] += m1[x+y*N] * m2[x*N+z];
      }
    }
  }
}
//multiplication of matrices in gpu
__global__ void mulMatrixGPU2D(long *MatA, long *MatB, long *MatC)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  long sum = 0;

  if (ix < N && iy < N)
  {
    for(int in =0;in<N;in++)
    {
        sum += MatA[ix*N+in] * MatB[in*N+iy];
    }
    MatC[ix*N+iy]=sum;
  }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = N;
    int ny = N;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(long);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    long *h_m1, *h_m2, *hostRef, *gpuRef;
    h_m1 = (long *)malloc(nBytes);
    h_m2 = (long *)malloc(nBytes);
    hostRef = (long *)malloc(nBytes);
    gpuRef = (long *)malloc(nBytes);

    // initialize data at host side

    fillMatrix(h_m1);
    fillMatrix(h_m2);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    mulMatrix(hostRef, h_m1, h_m2);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("sumMatrixOnHost elapsed %f ms\n", duration_ms.count());

    // malloc device global memory
    long *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_m1, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_m2, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = 2;
    int dimy = 256;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    start_cpu =  chrono::high_resolution_clock::now();
    mulMatrixGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;


    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // Compare results
    if(checkResult(hostRef, gpuRef))
      printf("They are equal\n");
    else
      printf("They are different\n");

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");


    // free host memory
    free(h_m1);
    free(h_m2);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
