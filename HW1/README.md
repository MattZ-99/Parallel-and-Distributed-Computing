<h1><center>Parallel and Distributed Computing Assignment 1<center/></h1>



​          姓名：张孟天					学号：517030910387



#### 实验环境

Visual Studio 2019 (v142), c++, 多线程调试(/Mtd)

#### Problem 1

Please write a program in OpenMP, to compute the sum of a vector,  $sum(x)=\sum_{i=0}^{n}x_i$.

Source code: *Parallel_lab1_vector_sum.cpp*

Serial implement function of vector sum.

~~~c++
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
~~~

Parallel function. using *parallel for* .

~~~c++
//并行函数， 使用parallel for
template <typename T>
T vector_sum_parallel(T* arr, int N) {
    T sum = 0;
    int i;
# pragma omp parallel for reduction(+: sum) private(i)
    for (i = 0; i < N; ++i) {
        sum += arr[i];
    }
    return sum;
}
~~~

Local parallel function. using *parallel* . 

- 这里做了一个手动分割的local function， 与之前 *parallel for* 的 结果进行对比。

~~~c++
//使用parallel，对local function 进行并行处理
# pragma omp parallel reduction(+: sum)
sum += vector_sum_local_parallel(arr, arr_len);


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
~~~



**效果验证**

我在实现了函数的基本功能之后，设计使用随机函数随机生成了$length = 100000000$的vector，分别使用不同的方式循行，并以串行的运行时间作为baseline，而串行的threads num设置为4，计算了不同程序的加速比。(*这里的parallel for 中的 schedule 使用的是default值)。

~~~
Serial             time: 242ms  sum: -53498     Speedup:1
Parallel (for)     time: 71ms   sum: -53498     Speedup:3.40845
Parallel (local)   time: 66ms   sum: -53498     Speedup:3.66667
~~~

实现了基本功能之后，我也尝试了一些（*parallel for* 的实现中）参数对运行速度以及加速比的影响。

- vector长度对加速比的影响（加速比计算为10次重复实验的平均值，$thread\_num=4$, schedule: default）。可以看到随着随着vector长度的增加，并行程序的加速比也会随之上升。一个合理的解释是，随着长度的增加，可以并发运行的程序增加，并行程序可以更好的利用多线程以加快运行速度，而相对，并行程序中的分发开销比例会降低。

  | Vector length | Speedup |
  | ------------- | ------- |
  | 5000000       | 2.4     |
  | 10000000      | 2.6     |
  | 50000000      | 3.26316 |
  | 100000000     | 3.40845 |
  | 200000000     | 3.56077 |

- 不同线程数对加速比的影响（$vector\_length=100000000$， schedule: default）。可以看到，随着并行线程数的增加，并行程序的加速比也会随之增加，但是加速比的增长速度并没有线程数增长的快。可能的原因是，在循环总次数一定的情况下，过多的线程会产生比较大的任务分发开销，从而导致加速比的提升速度并没有线程数快。

  ~~~
  thread_num = 2
  Serial             time: 238ms  sum: -53498     Speedup:1
  Parallel (for)     time: 116ms  sum: -53498     Speedup:2.05172
  Parallel (local)   time: 126ms  sum: -53498     Speedup:1.88889
  ~~~

  ~~~
  thread_num = 4
  Serial             time: 242ms  sum: -53498     Speedup:1
  Parallel (for)     time: 71ms   sum: -53498     Speedup:3.40845
  Parallel (local)   time: 66ms   sum: -53498     Speedup:3.66667
  ~~~

  ~~~
  thread_num = 8
  Serial             time: 245ms  sum: -53498     Speedup:1
  Parallel (for)     time: 48ms   sum: -53498     Speedup:5.10417
  Parallel (local)   time: 38ms   sum: -53498     Speedup:6.44737
  ~~~
- 不同的schedule参数对 *parallel for* 加速比的影响（$thread\_num=4$， $vector\_length=100000000$）。可以看到，不同的schedule方式确实会对加速比产生比较大的影响。同时不同的chunk大小也会对结果有比较大的影响。其中，过小的chunk值，会导致并行线程程序频繁的分配chunk（这个过程是单线程的），从而导致加速比很低。甚至，较为复杂的dynamic schedule方式会出现负优化的现象。

| schedule method         | speedup | schedule method          | speedup  | schedule method         | speedup |
| ----------------------- | ------- | ------------------------ | -------- | ----------------------- | ------- |
| schedule(static, 1)     | 2.6     | schedule(dynamic, 1)     | 0.109614 | schedule(guided, 1)     | 3.52174 |
| schedule(static, 2)     | 2.03448 | schedule(dynamic, 16)    | 0.737313 | schedule(guided, 16)    | 3.55882 |
| schedule(static, 4)     | 2.19643 | schedule(dynamic, 128)   | 1.62416  | schedule(guided, 64)    | 3.78125 |
| schedule(static, 8)     | 2.68539 | schedule(dynamic, 1024)  | 3.49275  | schedule(guided, 256)   | 3.71642 |
| schedule(static, 16)    | 2.63043 | schedule(dynamic, 2048)  | 3.72308  | schedule(guided, 1024)  | 3.55882 |
| schedule(static, 64)    | 2.52041 | schedule(dynamic, 4096)  | 3.68657  | schedule(guided, 65536) | 3.49296 |
| schedule(static, 1024)  | 3.0625  | schedule(dynamic, 65536) | 3.37333  |                         |         |
| schedule(static, 65536) | 3.32877 | schedule(runtime)        | 3.32432  |                         |         |






#### Problem 2

Please implement a function to compute matrix multiplication in OpenMP. You should consider the sizes of the matrices variable, $Mat_C=Mat_A\times Mat_B$.

Source code: *Parallel_lab1_matrix_multiplication.cpp* 

Serial implement function of matrix multiplication.

~~~c++
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
~~~

Parallel function. using *parallel for* .

~~~c++
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
~~~

Local parallel function. using *parallel* . 

~~~c++
//使用parallel，对local function 进行并行处理
#pragma omp parallel
Matrix_mul_parallel_local(Mat_A, Mat_B, Mat_C, M, N, K);

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
~~~

Parallel function. using *parallel for collapse(2)* .

~~~c++
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
~~~

Parallel function. using *parallel for, with matrix transpose first* . 首先对Mat_B进行转置以充分利用cache.

~~~c++
//并行实现的矩阵乘法 (parallel for, with 矩阵转置）
template <typename T>
void Matrix_mul_parallel_transpose(T** Mat_A, T** Mat_B, T** Mat_C, int M, int N, int K) {
	int i, j, k;
	T** Mat_B_T;
	malloc_Mat(Mat_B_T, N, M);
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
~~~

**效果验证**

在实现了基本的函数功能之后，我设计实现了随机函数来生成特定大小的矩阵作为验证数据，进行了一系列的效果验证。

~~~
thread_num = 4
matrix size. Mat_A: 500*500, Mat_B: 500*500
Serial              time: 653ms         Speedup:1
Parallel            time: 191ms         Speedup:3.41885
Parallel(collapse)  time: 188ms         Speedup:3.4734
Parallel(transpose) time: 143ms         Speedup:4.56643
Parallel (local)    time: 164ms         Speedup:3.98171
~~~

在此次实验过程中，我也做了一些额外的工作，来尝试提高运行速度。

- 如上实验结果所示，尝试使用local function手动进行循环划分，与parallel for 的程序进行对比。local function的手动划分方式确实可以在一定程度上减少循环任务划分所产生的开销。

- 如上实验结果所示，尝试*parallel for* 中的 collapse(2) 来展开两层循环进行并行操作，从而能在更细粒度进行并行化分。结果表明，确实能够一定程度上提高运行速度。

- 鉴于c++二维数组的横向储存方式，首先对矩阵$Mat\_B$进行转置操作，在进行循环，从而充分利用计算机cache，从而能够大幅提高运行速度，尤其对于较大的矩阵。

- 探索不同的schedule方式对于*parallel for* 的加速比影响。（$thread\_num = 4$, $Mat\_A: 500*500$, $Mat\_B: 500*500$)

  | schedule method        | speedup | schedule method         | speedup | schedule method | speedup |
  | ---------------------- | ------- | ----------------------- | ------- | --------------- | ------- |
  | schedule(static, 1)    | 3.07407 | schedule(dynamic, 1) | 3.82099 | schedule(guided, 1) | 3.71751 |
  | schedule(static, 128)  | 3.21693 | schedule(dynamic, 128) | 3.32402 | schedule(guided, 1) | 3.20745 |
  | schedule(static, 1024) | 0.85034 | schedule(dynamic, 1024) |  1.01975       | schedule(guided, 1) | 1.0162 |

  其中，当$chunk=1024$时，加速比非常小甚至负优化，是因为，此时，仅对一层for循环进行并行操作时，总的循环数量不足，从而无法利用到全部的threads。

- 探索矩阵大小对结果的影响。可见，越大的矩阵size，并行的提升效果越大。

  ~~~
  thread_num = 4
  matrix size. Mat_A: 100*100, Mat_B: 100*100
  Serial              time: 3ms   Speedup:1
  Parallel            time: 3ms   Speedup:1
  Parallel(collapse)  time: 1ms   Speedup:3
  Parallel(transpose) time: 1ms   Speedup:3
  Parallel (local)    time: 2ms   Speedup:1.5
  ~~~

  ~~~
  thread_num = 4
  matrix size. Mat_A: 500*500, Mat_B: 500*500
  Serial              time: 695ms         Speedup:1
  Parallel            time: 220ms         Speedup:3.15909
  Parallel(collapse)  time: 201ms         Speedup:3.45771
  Parallel(transpose) time: 155ms         Speedup:4.48387
  Parallel (local)    time: 209ms         Speedup:3.32536
  ~~~

  ~~~
  thread_num = 4
  matrix size. Mat_A: 1000*1000, Mat_B: 1000*1000
  Serial              time: 6784ms        Speedup:1
  Parallel            time: 1892ms        Speedup:3.58562
  Parallel(collapse)  time: 1905ms        Speedup:3.56115
  Parallel(transpose) time: 933ms         Speedup:7.27117
  Parallel (local)    time: 1880ms        Speedup:3.60851
  ~~~

  ~~~
  thread_num = 4
  matrix size. Mat_A: 100*1000, Mat_B: 1000*1000
  Serial              time: 779ms         Speedup:1
  Parallel            time: 229ms         Speedup:3.40175
  Parallel(collapse)  time: 221ms         Speedup:3.52489
  Parallel(transpose) time: 118ms         Speedup:6.6017
  Parallel (local)    time: 244ms         Speedup:3.19262
  ~~~

- 探索线程数对结果的影响。可以看到，随着并行线程数的增加，并行程序的加速比也会随之增加，但是加速比的增长速度并没有线程数增长的快。可能的原因是，在循环总次数较小的情况下，过多的线程会产生比较大的任务分发开销，从而导致加速比无法继续增长。

  ~~~
  thread_num = 2
  matrix size. Mat_A: 500*500, Mat_B: 500*500
  Serial              time: 479ms         Speedup:1
  Parallel            time: 300ms         Speedup:1.59667
  Parallel(collapse)  time: 278ms         Speedup:1.72302
  Parallel(transpose) time: 231ms         Speedup:2.07359
  Parallel (local)    time: 312ms         Speedup:1.53526
  ~~~

  ~~~
  thread_num = 4
  matrix size. Mat_A: 500*500, Mat_B: 500*500
  Serial              time: 653ms         Speedup:1
  Parallel            time: 191ms         Speedup:3.41885
  Parallel(collapse)  time: 188ms         Speedup:3.4734
  Parallel(transpose) time: 143ms         Speedup:4.56643
  Parallel (local)    time: 164ms         Speedup:3.98171
  ~~~

  ~~~
  thread_num = 8
  matrix size. Mat_A: 500*500, Mat_B: 500*500
  Serial              time: 523ms         Speedup:1
  Parallel            time: 158ms         Speedup:3.31013
  Parallel(collapse)  time: 145ms         Speedup:3.6069
  Parallel(transpose) time: 133ms         Speedup:3.93233
  Parallel (local)    ti‘me: 127ms         Speedup:4.11811
  ~~~

  ~~~
  thread_num = 8
  matrix size. Mat_A: 1000*1000, Mat_B: 1000*1000
  Serial              time: 6752ms        Speedup:1
  Parallel            time: 1315ms        Speedup:5.1346
  Parallel(collapse)  time: 1300ms        Speedup:5.19385
  Parallel(transpose) time: 636ms         Speedup:10.6164
  Parallel (local)    time: 1370ms        Speedup:4.92847
  ~~~

  

