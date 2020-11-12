#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <ctime>
#ifdef _OPENMP
# include <omp.h>
#endif

using namespace std;

//**********************************************************
//本文件是一个并行计算vector的元素和的项目
//测试输入是一个，有随机函数生成的整数向量
//核心部分包括，vector_sum_serial， vector_sum_parallel，vector_sum_local_parallel（利用local function）
//**********************************************************

// 函数声明*********

void Random(int* a, int N, int l, int r); //生成范围在l~r的随机数 

void Print(int* a, int N);  //输出向量序列

template <typename T> T vector_sum_serial(T* arr, int N); //串行函数

template <typename T> T vector_sum_parallel(T* arr, int N);//并行函数， 使用parallel for

template <typename T> T vector_sum_local_parallel(T* arr, int N);//局部函数，运行时使用parallel进行函数并行处理

//******************

int main(int argc, char* argv[]) {

	int thread_count = strtol(argv[1], NULL, 10);
	omp_set_num_threads(thread_count); //设置线程数


    //随机生成验证数据
    int arr_len = 100000000;
    int* arr = new int[arr_len];
    Random(arr, arr_len, -5, 5);
    //Print(arr, arr_len);


    clock_t time_start, time_end;
    int sum;
    double speedup;


    //串行计算 baseline
    time_start = clock();
    sum = vector_sum_serial(arr, arr_len);
    time_end = clock();

    int time_serial = time_end - time_start;
    speedup = (float)time_serial / (float)time_serial;
    printf("%s time: %dms\tsum: %d\tSpeedup:%g\n", "Serial            ", time_serial, sum, speedup);


    //使用 parallel for 并行训练
    time_start = clock();
    sum = vector_sum_parallel(arr, arr_len);
    time_end = clock();

    int time_parallel = time_end - time_start;
    speedup = (float)time_serial / (float)time_parallel;
    printf("%s time: %dms\tsum: %d\tSpeedup:%g\n", "Parallel (for)    ", time_parallel, sum, speedup);


    //使用parallel，对local function 进行并行处理
    sum = 0;
    time_start = clock();
    # pragma omp parallel reduction(+: sum)
    sum += vector_sum_local_parallel(arr, arr_len);
    time_end = clock();

    int time_parallel_local = time_end - time_start;
    speedup = (float)time_serial / (float)time_parallel_local;
    printf("%s time: %dms\tsum: %d\tSpeedup:%g\n", "Parallel (local)  ", time_parallel_local, sum, speedup);

	return 0;
}


// 函数定义*********

void Random(int* a, int N, int l, int r)//生成范围在l~r的随机数 
{
    srand(100);  //设置时间种子
    for (int i = 0; i < N; i++) {
        a[i] = rand() % (r - l + 1) + l;//生成区间l~r的随机数 
    }
}

void Print(int* a, int N)
{
    for (int i = 0; i < N; i++)
        cout << a[i] << " ";
    cout << endl;
}

//串行函数
template <typename T>
T vector_sum_serial(T* arr, int N) {
    T sum = 0;
    int i;
    for (i = 0; i < N; i++) {
        sum += arr[i];
    }
    return sum;
}

//并行函数， 使用parallel for
template <typename T>
T vector_sum_parallel(T* arr, int N) {
    T sum = 0;
    int i;
# pragma omp parallel for reduction(+: sum) private(i) schedule(dynamic, 2048)
    for (i = 0; i < N; ++i) {
        sum += arr[i];
    }
    return sum;
}

//局部函数，运行时使用parallel进行函数并行处理
//根据线程号以及总线程数，手动对循环进行切割
template <typename T>
T vector_sum_local_parallel(T* arr, int N) {
    T local_result = 0;
    int local_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    int a = local_rank * N / thread_count;
    int b = (local_rank + 1) * N / thread_count;
    for (int i = a; i < b; i++) {
        local_result += arr[i];
    }
    return local_result;
}