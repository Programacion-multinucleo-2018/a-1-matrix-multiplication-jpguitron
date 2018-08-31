
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <chrono>

#include <iostream>

#include <omp.h>

#define N 1000

using namespace std;

void fillMatrix(long * matrix)
{
  int i;
  int size = N*N;
  for(i = 0; i < size; i++)
  {
      matrix[i] = i;
  }
}

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

void mulMatrix(long * m_r, long * m1, long * m2)
{
  int threadID, totalThreads;

    int x;
    int y;
    int z;

    //Declaring that it will run parallel
    #pragma omp parallel for private(x, y, z) shared(m_r, m1, m2)
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

int main()
{
  int nx = N;
  int ny = N;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(long);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // malloc host memory
  long *h_m_r, *h_m1, *h_m2;
  h_m1 = (long *)malloc(nBytes);
  h_m2 = (long *)malloc(nBytes);
  h_m_r = (long *)malloc(nBytes);

  //Fill the matrix
  fillMatrix(h_m1);
  fillMatrix(h_m2);

  omp_set_num_threads(4);

  auto start_cpu =  chrono::high_resolution_clock::now();
  mulMatrix(h_m_r, h_m1, h_m2);
  auto end_cpu =  chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  printf("elapsed %f ms\n", duration_ms.count());

  //Free arrays memory
  free(h_m1);
  free(h_m2);
  free(h_m_r);


}
