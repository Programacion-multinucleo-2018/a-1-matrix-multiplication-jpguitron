#include <cstdlib>
#include <cstdio>
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

//multiplication of matrices in cpu with threads
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

int main()
{
  int nx = N; //The size of the Matrix
  int ny = N;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(long);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // malloc host memory
  long *h_m_r, *h_m1, *h_m2;
  h_m1 = (long *)malloc(nBytes);
  h_m2 = (long *)malloc(nBytes);
  h_m_r = (long *)malloc(nBytes);

  fillMatrix(h_m1);
  fillMatrix(h_m2);

  auto start_cpu =  chrono::high_resolution_clock::now();
  mulMatrix(h_m_r, h_m1, h_m2);
  auto end_cpu =  chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  printf("elapsed %f ms\n", duration_ms.count());

  //Free arrays memory
  free(h_m1);
  free(h_m2);
  free(h_m_r);

  return 0;
}
