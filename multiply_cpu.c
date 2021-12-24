#include<stdio.h>
#include<time.h>
int main()
{
    clock_t start,end;  
    start = clock();  
    int size=8192;
    int eRSA=65537;
    int count=0;
    for(int i=0;i<size;i++){
        unsigned long long mat1=rand()+1;
        unsigned long long mat2=rand()+1;
        
        unsigned long long temp=1;
        for(int j=0;j<mat2;j++){
            temp*=mat1;
        }
        unsigned long long result=temp%eRSA;
        //printf("mat1=%llu mat2=%llu\n",mat1,mat2);
        // printf("result=%llu\n",result);
        // printf("\n");
        // count++;
        // printf("count=%d\n",count);
        
    }
    
    end = clock();
    printf("%f\n",(double)(end-start)/CLOCKS_PER_SEC);
}