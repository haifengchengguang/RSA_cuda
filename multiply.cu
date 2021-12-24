/*
multiply.cu
nvcc multiply.cu -o multiply
*/
#include <stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
// #include<iostream>
// using namespace std;
#define BLOCK_NUM 32   //块数量
#define THREAD_NUM 256 // 每个块中的线程数
#define R_SIZE BLOCK_NUM * THREAD_NUM
#define M_SIZE R_SIZE * R_SIZE
//#define M_SIZE 10
// int powInt(int a,int b){
//     if(a==0&&b!=0){return 0;}
//     else if (a==1)
//     {
//         return 1;
//     }
//     else if (b==0)
//     {
//         return a;
//     }
//     else{
//     int result=1;
//     for(int i=0;i<b;i++){
//         result*=a;
//     }
//     return result;
//     }
// }
__global__ void mat_mul(unsigned long long *mat1, unsigned long long *mat2,int eRSA, unsigned long long *result) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
   
    const int row = bid * THREAD_NUM + tid;
    // for (int c = 0; c < R_SIZE; c++) {
    //     for (int n = 0; n < R_SIZE; n++) {
    //         result[row*R_SIZE+c] += mat1[row*R_SIZE+n] * mat2[n*R_SIZE+c];
    //     }
    // }
    for(int i=0;i<R_SIZE;i++){
        //result[row*R_SIZE+i]=mat1[row*R_SIZE+i]*mat2[row*R_SIZE+i];
        int temp=1;
        for(int j=0;j<mat2[row*R_SIZE+i];j++)
        {
            temp*=mat1[row*R_SIZE+i];
        }
        result[row*R_SIZE+i]=temp%eRSA;
    }
}

int main(int argc, char *argv[]) {
    clock_t start,end;  
    start = clock();  
    // time_t start,end;  
    // start =time(NULL);//or time(&start);  
    int eRSA=65537;
    unsigned long long *mat1, *mat2, *result;
    unsigned long long *g_mat1, *g_mat2, *g_mat_result;
    
    // 用一位数组表示二维矩阵
    mat1 = (unsigned long long*) malloc(M_SIZE * sizeof(unsigned long long));
    mat2 = (unsigned long long*) malloc(M_SIZE * sizeof(unsigned long long));
    //eRSA = (int*) malloc(M_SIZE * sizeof(int));
    result = (unsigned long long*) malloc(M_SIZE * sizeof(unsigned long long));

    // initialize
    for (int i = 0; i < M_SIZE; i++) {
        mat1[i] = rand()+1;
        mat2[i] = rand()+1;
        //eRSA[i]=65537;
        result[i] = 0;
        
    }

    cudaMalloc((void **)&g_mat1, sizeof(unsigned long long) * M_SIZE);
    cudaMalloc((void **)&g_mat2, sizeof(unsigned long long) * M_SIZE);
    //cudaMalloc((void **)&g_eRSA, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_mat_result, sizeof(unsigned long long) * M_SIZE);

    cudaMemcpy(g_mat1, mat1, sizeof(unsigned long long) * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat2, mat2, sizeof(unsigned long long) * M_SIZE, cudaMemcpyHostToDevice);
    //cudaMemcpy(g_eRSA, eRSA, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
    mat_mul<<<BLOCK_NUM, THREAD_NUM>>>(g_mat1, g_mat2,eRSA, g_mat_result);

    cudaMemcpy(result, g_mat_result, sizeof(unsigned long long) * M_SIZE, cudaMemcpyDeviceToHost);
    //…calculating…  
    // end =time(NULL);  
    // printf("time=%f\n",difftime(end,start));  
    end = clock();  
    printf("time=%f\n",(double)(end-start)/CLK_TCK);  
    printf("sizeof(unsigned long long)=%zd",sizeof(unsigned long long));
    // for(int i=0;i<R_SIZE;i++)
    // {
    //     printf("mat1[%d]=%lld\n",i,mat1[i]);
    //     printf("mat2[%d]=%lld\n",i,mat2[i]);
    //     printf("result[%d]=%lld\n",i,result[i]);
    //     printf("-------------\n");
    // }

}