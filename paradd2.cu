// parallel add of large integers
// requires CC 2.0 or higher
// compile with:
// nvcc -O3 -arch=sm_20 -o paradd2 paradd2.cu
#include <stdio.h>
#include <stdlib.h>

#define MAXSIZE 1024 // the number of 64 bit quantities that can be added
#define LLBITS 64  // the number of bits in a long long
#define BSIZE ((MAXSIZE + LLBITS -1)/LLBITS) // MAXSIZE when packed into bits
#define nTPB MAXSIZE

// define either GPU or GPUCOPY, not both -- for timing
#define GPU
//#define GPUCOPY

#define LOOPCNT 1000

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// perform c = a + b, for unsigned integers of psize*64 bits.
// all work done in a single threadblock.
// multiple threadblocks are handling multiple separate addition problems
// least significant word is at a[0], etc.

__global__ void paradd(const unsigned size, const unsigned psize, unsigned long long *c, const unsigned long long *a, const unsigned long long *b){

  __shared__ unsigned long long carry_through[BSIZE];
  __shared__ unsigned long long carry[BSIZE+1];
  __shared__ volatile unsigned mcarry;
  __shared__ volatile unsigned mcarry_through;

  unsigned idx = threadIdx.x + (psize * blockIdx.x);
  if ((threadIdx.x < psize) && (idx < size)){
    // handle 64 bit unsigned add first
    unsigned long long cr1 = a[idx];
    unsigned long long lc = cr1 + b[idx];
    // handle carry
    if (threadIdx.x < BSIZE){
      carry[threadIdx.x] = 0;
      carry_through[threadIdx.x] = 0;
      }
    if (threadIdx.x == 0){
      mcarry = 0;
      mcarry_through = 0;
      }
    __syncthreads();
    if (lc < cr1){
      if ((threadIdx.x%LLBITS) != (LLBITS-1))  
        atomicAdd(&(carry[threadIdx.x/LLBITS]), (2ull<<(threadIdx.x%LLBITS)));
      else atomicAdd(&(carry[(threadIdx.x/LLBITS)+1]), 1);
      }
    // handle carry-through
    if (lc == 0xFFFFFFFFFFFFFFFFull) 
      atomicAdd(&(carry_through[threadIdx.x/LLBITS]), (1ull<<(threadIdx.x%LLBITS))); 
    __syncthreads();
    if (threadIdx.x < ((psize + LLBITS-1)/LLBITS)){
      // only 1 warp executing within this if statement
      unsigned long long cr3 = carry_through[threadIdx.x];
      cr1 = carry[threadIdx.x] & cr3;
      // start of sub-add
      unsigned long long cr2 = cr3 + cr1;
      if (cr2 < cr1) atomicAdd((unsigned *)&mcarry, (2u<<(threadIdx.x)));
      if (cr2 == 0xFFFFFFFFFFFFFFFFull) atomicAdd((unsigned *)&mcarry_through, (1u<<threadIdx.x));
      if (threadIdx.x == 0) {
        unsigned cr4 = mcarry & mcarry_through;
        cr4 += mcarry_through;
        mcarry |= (mcarry_through ^ cr4); 
        }
      if (mcarry & (1u<<threadIdx.x)) cr2++;
      // end of sub-add
      carry[threadIdx.x] |= (cr2 ^ cr3);
      }
    __syncthreads();
    if (carry[threadIdx.x/LLBITS] & (1ull<<(threadIdx.x%LLBITS))) lc++;
    c[idx] = lc;
  }
}

int main() {

  unsigned long long *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *c;
  unsigned at_once = 256;   // valid range = 1 .. 65535
  unsigned prob_size = MAXSIZE ; // valid range = 1 .. MAXSIZE
  unsigned dsize = at_once * prob_size;
  cudaEvent_t t_start_gpu, t_start_cpu, t_end_gpu, t_end_cpu;
  float et_gpu, et_cpu, tot_gpu, tot_cpu;
  tot_gpu = 0;
  tot_cpu = 0;


  if (sizeof(unsigned long long) != (LLBITS/8)) {printf("Word Size Error\n"); return 1;}
  if ((c = (unsigned long long *)malloc(dsize * sizeof(unsigned long long)))  == 0) {printf("Malloc Fail\n"); return 1;}

  cudaHostAlloc((void **)&h_a, dsize * sizeof(unsigned long long), cudaHostAllocDefault);
  cudaCheckErrors("cudaHostAlloc1 fail");
  cudaHostAlloc((void **)&h_b, dsize * sizeof(unsigned long long), cudaHostAllocDefault);
  cudaCheckErrors("cudaHostAlloc2 fail");
  cudaHostAlloc((void **)&h_c, dsize * sizeof(unsigned long long), cudaHostAllocDefault);
  cudaCheckErrors("cudaHostAlloc3 fail");

  cudaMalloc((void **)&d_a, dsize * sizeof(unsigned long long));
  cudaCheckErrors("cudaMalloc1 fail");
  cudaMalloc((void **)&d_b, dsize * sizeof(unsigned long long));
  cudaCheckErrors("cudaMalloc2 fail");
  cudaMalloc((void **)&d_c, dsize * sizeof(unsigned long long));
  cudaCheckErrors("cudaMalloc3 fail");
  cudaMemset(d_c, 0, dsize*sizeof(unsigned long long));

  cudaEventCreate(&t_start_gpu);
  cudaEventCreate(&t_end_gpu);
  cudaEventCreate(&t_start_cpu);
  cudaEventCreate(&t_end_cpu);

  for (unsigned loops = 0; loops <LOOPCNT; loops++){
  //create some test cases
  if (loops == 0){
  for (int j=0; j<at_once; j++)
  for (int k=0; k<prob_size; k++){
    int i= (j*prob_size) + k;
    h_a[i] = 0xFFFFFFFFFFFFFFFFull;
    h_b[i] = 0;
    }
    h_a[prob_size-1] = 0;
    h_b[prob_size-1] = 1;
    h_b[0] = 1;
  }
  else if (loops == 1){
  for (int i=0; i<dsize; i++){
    h_a[i] = 0xFFFFFFFFFFFFFFFFull;
    h_b[i] = 0;
    }
    h_b[0] = 1;
  }
  else if (loops == 2){
  for (int i=0; i<dsize; i++){
    h_a[i] = 0xFFFFFFFFFFFFFFFEull;
    h_b[i] = 2;
    }
    h_b[0] = 1;
  }
  else {
  for (int i = 0; i<dsize; i++){
    h_a[i] = (((unsigned long long)rand())<<33) + (unsigned long long)rand();
    h_b[i] = (((unsigned long long)rand())<<33) + (unsigned long long)rand();
    }
  }
#ifdef GPUCOPY
  cudaEventRecord(t_start_gpu, 0);
#endif
  cudaMemcpy(d_a, h_a, dsize*sizeof(unsigned long long), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy1 fail");
  cudaMemcpy(d_b, h_b, dsize*sizeof(unsigned long long), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy2 fail");
#ifdef GPU
  cudaEventRecord(t_start_gpu, 0);
#endif
  paradd<<<at_once, nTPB>>>(dsize, prob_size, d_c, d_a, d_b);
  cudaCheckErrors("Kernel Fail");
#ifdef GPU
  cudaEventRecord(t_end_gpu, 0);
#endif
  cudaMemcpy(h_c, d_c, dsize*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy3 fail");
#ifdef GPUCOPY
  cudaEventRecord(t_end_gpu, 0);
#endif
  cudaEventSynchronize(t_end_gpu);
  cudaEventElapsedTime(&et_gpu, t_start_gpu, t_end_gpu);
  tot_gpu += et_gpu;
  cudaEventRecord(t_start_cpu, 0);
  //also compute result on CPU for comparison
  for (int j=0; j<at_once; j++) {
    unsigned rc=0;
    for (int n=0; n<prob_size; n++){
      unsigned i = (j*prob_size) + n;
      c[i] = h_a[i] + h_b[i];
      if (c[i] < h_a[i]) {
        c[i] += rc;
        rc=1;}
      else {
        if ((c[i] += rc) != 0) rc=0;
        }
      if (c[i] != h_c[i]) {printf("Results mismatch at offset %d, GPU = 0x%lX, CPU = 0x%lX\n", i, h_c[i], c[i]); return 1;}
      }
    }
  cudaEventRecord(t_end_cpu, 0);
  cudaEventSynchronize(t_end_cpu);
  cudaEventElapsedTime(&et_cpu, t_start_cpu, t_end_cpu);
  tot_cpu += et_cpu;
  if ((loops%(LOOPCNT/10)) == 0) printf("*\n");
  }
  printf("\nResults Match!\n");
  printf("Average GPU time = %fms\n", (tot_gpu/LOOPCNT));
  printf("Average CPU time = %fms\n", (tot_cpu/LOOPCNT));

  return 0;
}