#include <stdio.h>
#include <stdlib.h>
#include <vector>
#ifdef _OPENMP
# include <omp.h>
#endif
#include< ctime >
#include <iostream>
using namespace std;

#include <iostream>

//**********************************************************
//本文件是一个并行处理矩阵乘法的程序
//测试数据为随机生成的矩阵
//测试的方式包括Matrix_mul_serial，Matrix_mul_parallel，Matrix_mul_parallel_local，Matrix_mul_parallel_collapse，Matrix_mul_parallel_transpose
//以及schedule的方式以chunk大小，并行threads数
//**********************************************************




// 函数声明*********

void malloc_Mat(int**& a, int xDim, int yDim);  // 初始化矩阵，给矩阵分配空间

void Random(int** Mat_A, int** Mat_B, int M, int N, int K, int l, int r);  //随机生成矩阵Mat_A, Mat_B

void Mat_zero(int** Mat, int M, int N);  //为矩阵Mat_C赋初值0

void Print_Mat(int**& Mat, int M, int N); //输出Mat

template <typename T> void Matrix_mul_serial(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K); //串行实现的矩阵乘法

template <typename T> void Matrix_mul_parallel(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K); //并行实现的矩阵乘法 （parallel for)

template <typename T> void Matrix_mul_parallel_local(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K); //并行实现的矩阵乘法 (使用 local function）

template <typename T> void Matrix_mul_parallel_collapse(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K); //并行实现的矩阵乘法 （parallel for collapse(2))

template <typename T> void Matrix_mul_parallel_transpose(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K); //并行实现的矩阵乘法 (parallel for, with 矩阵转置）

template <typename T> void Matrix_transpose(T** Mat, T** Mat_T, int M, int N); //矩阵转置函数

//******************



int main(int argc, char* argv[]) {

	int thread_count = strtol(argv[1], NULL, 10);
	omp_set_num_threads(thread_count); //设置线程数
	printf("thread_num = %d\n", thread_count);

	int M = 500, N = 500, K = 500;  //设置矩阵大小 Mat_A:M*N, Mat_B:N*K
	int** Mat_A, ** Mat_B, ** Mat_C;
	printf("matrix size. Mat_A: %d*%d, Mat_B: %d*%d\n", M, N, N, K);

	//分配空间
	malloc_Mat(Mat_A, M, N);
	malloc_Mat(Mat_B, N, K);
	malloc_Mat(Mat_C, M, K);

	//随机生成矩阵
	Random(Mat_A, Mat_B, M, N, K, 1, 10);
	//Mat_zero(Mat_C, M, K);
	//Print_Mat(Mat_A, M, N);
	//Print_Mat(Mat_B, N, K);
	//Print_Mat(Mat_C, M, K);


	clock_t time_start, time_end;
	int sum;
	double speedup;

	//串行矩阵乘法
	time_start = clock();
	Matrix_mul_serial(Mat_A, Mat_B, Mat_C, M, N, K);
	time_end = clock();

	int time_serial = time_end - time_start;
	speedup = (float)time_serial / (float)time_serial;
	printf("%s time: %dms \tSpeedup:%g\n", "Serial             ", time_serial, speedup);
	
	//parallel for
	time_start = clock();
	Matrix_mul_parallel(Mat_A, Mat_B, Mat_C, M, N, K);
	time_end = clock();

	int time_parallel = time_end - time_start;
	speedup = (float)time_serial / (float)time_parallel;
	printf("%s time: %dms \tSpeedup:%g\n", "Parallel           ", time_parallel, speedup);
	
	//parallel for collapse(2)
	time_start = clock();
	Matrix_mul_parallel_collapse(Mat_A, Mat_B, Mat_C, M, N, K);
	time_end = clock();

	int time_parallel_collapse = time_end - time_start;
	speedup = (float)time_serial / (float)time_parallel_collapse;
	printf("%s time: %dms \tSpeedup:%g\n", "Parallel(collapse) ", time_parallel_collapse, speedup);

	//parallel for, with matrix transpose, 首先对Mat_B进行转置以充分利用cache
	time_start = clock();
	Matrix_mul_parallel_transpose(Mat_A, Mat_B, Mat_C, M, N, K);
	time_end = clock();

	int time_parallel_transpose = time_end - time_start;
	speedup = (float)time_serial / (float)time_parallel_transpose;
	printf("%s time: %dms \tSpeedup:%g\n", "Parallel(transpose)", time_parallel_transpose, speedup);


	time_start = clock();
	//使用parallel，对local function 进行并行处理
	#pragma omp parallel
	Matrix_mul_parallel_local(Mat_A, Mat_B, Mat_C, M, N, K);
	time_end = clock();

	int time_parallel_local = time_end - time_start;
	speedup = (float)time_serial / (float)time_parallel_local;
	printf("%s time: %dms \tSpeedup:%g\n", "Parallel (local)   ", time_parallel_local, speedup);
	

	return 0;
}



// 函数定义*********

// 初始化矩阵，给矩阵分配空间
void malloc_Mat(int**& mat, int xDim, int yDim) {
	mat= new int* [xDim];
	for (int i = 0; i < xDim; i++)
		mat[i] = new int[yDim];
}

//随机生成矩阵Mat_A, Mat_B
void Random(int** Mat_A, int** Mat_B, int M, int N, int K, int l, int r)
{
	srand(10);  //设置时间种子
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			Mat_A[i][j] = rand() % (r - l + 1) + l;
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < K; j++) {
			Mat_B[i][j] = rand() % (r - l + 1) + l;
		}
	}
}

//为矩阵Mat_C赋初值0
void Mat_zero(int** Mat, int M, int N)
{
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			Mat[i][j] = 0;
		}
	}
}

//输出Mat
void Print_Mat(int**& Mat, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout << Mat[i][j] << "\t";
		}
		cout << "\n" << endl;
	}
	cout << "\n" << endl;
}

//串行实现的矩阵乘法
template <typename T>
void Matrix_mul_serial(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K) {
	int i, j, k;
	for (i = 0; i < M; i++) {
		for (k = 0; k < K; k++) {
			T local_res = 0;
			for (j = 0; j < N; j++) {
				local_res += Mat_A[i][j] * Mat_B[j][k];
			}
			Mat_C[i][k] = local_res;
		}
	}
}

//并行实现的矩阵乘法 （parallel for)
template <typename T>
void Matrix_mul_parallel(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K) {
	int i, j, k;
	# pragma omp parallel for private(i, j, k) schedule(runtime)
	for (i = 0; i < M; i++) {
		for (k = 0; k < K; k++) {
			T local_res = 0;
			for (j = 0; j < N; j++) {
				local_res += Mat_A[i][j] * Mat_B[j][k];
			}
			Mat_C[i][k] = local_res;
		}
	}
}

// 并行实现的矩阵乘法(使用 local function）
template <typename T>
void Matrix_mul_parallel_local(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K) {
	int local_rank = omp_get_thread_num();
	int thread_count = omp_get_num_threads();
	int pl = local_rank * M * K / thread_count;
	int pr = (local_rank + 1) * M * K / thread_count;
	int m, k;
	for (int i = pl; i < pr; i++) {
		m = i / K;
		k = i % K;
		T local_res = 0;
		for (int n = 0; n < N; n++) {
			local_res += Mat_A[m][n] * Mat_B[n][k];
		}
		Mat_C[m][k] = local_res;
	}
}

//并行实现的矩阵乘法 (parallel for, with 矩阵转置）
template <typename T>
void Matrix_mul_parallel_transpose(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K) {
	int i, j, k;
	T** Mat_B_T;
	malloc_Mat(Mat_B_T, K, N);
	Matrix_transpose(Mat_B, Mat_B_T, N, K);
	# pragma omp parallel for private(i, j, k) collapse(2) schedule(runtime)
	for (i = 0; i < M; i++) {
		for (k = 0; k < K; k++) {
			T local_res = 0;
			for (j = 0; j < N; j++) {
				local_res += Mat_A[i][j] * Mat_B_T[k][j];
			}
			Mat_C[i][k] = local_res;
		}
	}
}

//矩阵转置函数
template <typename T>
void Matrix_transpose(T** Mat, T** Mat_T, int M, int N) {
	# pragma omp parallel for collapse(2) schedule(runtime)
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			Mat_T[j][i] = Mat[i][j];
		}
	}
}

//并行实现的矩阵乘法 （parallel for collapse(2))
template <typename T>
void Matrix_mul_parallel_collapse(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K) {
	int i, j, k;
# pragma omp parallel for private(i, j, k) collapse(2) schedule(runtime)
	for (i = 0; i < M; i++) {
		for (k = 0; k < K; k++) {
			T local_res = 0;
			for (j = 0; j < N; j++) {
				local_res += Mat_A[i][j] * Mat_B[j][k];
			}
			Mat_C[i][k] = local_res;
		}
	}
}

