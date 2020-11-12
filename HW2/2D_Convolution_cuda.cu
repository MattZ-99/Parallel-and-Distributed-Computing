# include <stdio.h>
# include <iostream>
# include <ctime>

# include "cuda_runtime.h"
# include "device_launch_parameters.h"

using namespace std;

//feature map [N, F, H, W]
//kernel [F, C, K, K]
//output tensor [N, F, H_, W_]
#define N 8
#define C 64
#define H 128
#define W 128
#define F 128
#define K 3
#define H_ (H-K+1)
#define W_ (W-K+1)
#define TILE_WIDTH 32
#define L C * K * K

// ***************************************************************************************
// 在此次实验中，我尝试实现了2D卷积程序
// 尝试了在cpu， gpu上的不同编程方式
// 在cpu上设计了不同的threads数，来观察对运算速度的影响
// 最后，也将卷积运算更改为了矩阵乘法的操作，并在shared memory上运行，从而对performance进行优化
// ***************************************************************************************

// 需要注意的是，
// 1.我将所有尝试的操作封装为一个个函数，放在main()函数中，然后先后执行，但是部分单线程的函数可能会
// 耗费大量的时间，如果，不需要，可以在main()中注释掉费时的操作。
//      conv_cpu(): Time use: 91555.2ms
//      conv_gpu_single(): Time use: 231721ms
//      conv_gpu_sample(): Time use: 32302.9ms
//      conv_gpu_channel(): Time use: 604.579ms
//      conv_gpu_col(): Time use: 131.125ms
//      conv_gpu_matrix_single_thread(): Time use: 296933ms
//      conv_gpu_matrix_channel(): Time use: 4218.25ms
//      conv_gpu_matrix_col(): Time use: 3390.3ms
//      conv_gpu_matrix_shared(): Time use: 3418.78ms

// 2. 我在这里写的 time use 时间均是整个运算过程的时间，其中包括了 cudaMemcpy, im2col, 以及gpu进行
// 计算的全部用时。我在报告中，对比的用时，会用到 nvprof 显示的仅 gpu/cpu 进行数据部分计算的时间。

// ******************************************
// ** nvcc 2D_Convolution_cuda.cu -o conv  **
// ** nvprof ./conv  or ./conv             **
// ** (nvprof 用来显示程序具体运行时间时使用) **
// ******************************************

// 函数声明
void conv_cpu(float * feature_map, float * kernel, float * output);  // cpu版本
void conv_gpu_single(float *feature_map, float *kernel, float *output); // gpu 单线程 
void conv_gpu_sample(float *feature_map, float *kernel, float *output); // gpu sample (N) 并行
void conv_gpu_channel(float *feature_map, float *kernel, float *output); // gpu channel (N, F) 并行
void conv_gpu_col(float *feature_map, float *kernel, float *output); // gpu block (N, F, H_, W_) 并行 
void conv_gpu_matrix_single_thread(float *feature_map, float *kernel, float *output); // gpu, im2col, 单线程
void conv_gpu_matrix_channel(float *feature_map, float *kernel, float *output); // gpu, im2col, channel (N, F) 并行
void conv_gpu_matrix_col(float *feature_map, float *kernel, float *output);  // gpu, im2col, block (N, F, H_, W_) 并行 
void conv_gpu_matrix_shared(float *feature_map, float *kernel, float *output); // gpu, im2col + shared memory, block (N, F, H_, W_) 并行 
bool result_is_equal(float * tensor1, float * tensor2); // 与cpu版本对比，检查结果是否正确
// 以下是对应的kernel函数
__global__ void conv_gpu_single_kernel(float *feature_map, float *kernel, float *output);
__global__ void conv_gpu_sample_kernel(float *feature_map, float *kernel, float *output);
__global__ void conv_gpu_channel_kernel(float *feature_map, float *kernel, float *output);
__global__ void conv_gpu_kernel_matrix_single_thread(float *feature_map_gpu, float *kernel_gpu, float *output_gpu);
__global__ void conv_gpu_kernel_matrix_channel(float *feature_map_gpu, float *kernel_gpu, float *output_gpu);
__global__ void conv_gpu_kernel_matrix_col(float *feature_map_gpu, float *kernel_gpu, float *output_gpu);
__global__ void conv_gpu_col_kernel(float *feature_map, float *kernel, float *output) ;
__global__ void conv_gpu_kernel_matrix_shared(float *feature_map_gpu, float *kernel_gpu, float *output_gpu);




int main(void) {
    
    //数据初始化
    float *feature_map, *kernel, *output, *output_cpu;
    long size_fm = N * C * H * W;
    long size_kernel = F * C * K * K;
    long size_out = N * F * H_ * W_;

    feature_map = (float *) malloc(size_fm * sizeof(float));
    kernel = (float *) malloc(size_kernel * sizeof(float));
    output = (float *) malloc(size_out * sizeof(float));
    output_cpu = (float *) malloc(size_out * sizeof(float));


    // input 赋初值的 部分， 尝试 三种赋初值的方法

    // for (int i = 0; i < size_fm; i++) feature_map[i] = 1.0;
    // for (int i = 0; i < size_kernel; i++) kernel[i] = 1.0;
    // for (int i = 0; i < size_out; i++) output[i] = 0.0;
    
    // feature map 设置 初值
    for (int i = 0; i < size_fm; ++i) feature_map[i] = i % 5 * 1.0;
    // kernel 赋初值
    for (int i = 0; i < size_kernel; ++i) kernel[i] = i % 3 * 1.0;
    
    //随机赋值
    // for (int i = 0; i < size_fm; ++i) feature_map[i] = 1.0 * (rand() % 200 -100) / 50;
    // for (int i = 0; i < size_kernel; ++i) kernel[i] = 1.0 * (rand() % 200 -100) / 50;


    clock_t time_start, time_end;
    bool correctness;
    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on cpu");
    time_start = clock();
    conv_cpu(feature_map, kernel, output_cpu);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output_cpu);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");


    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on gpu, using single thread");
    time_start = clock();
    conv_gpu_single(feature_map, kernel, output);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");

    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on gpu, parallel on sample level (1 block, N threads)");
    time_start = clock();
    conv_gpu_sample(feature_map, kernel, output);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");

    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on gpu, parallel on channel level (N blocks, F threads)");
    time_start = clock();
    conv_gpu_channel(feature_map, kernel, output);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");

    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on gpu, parallel on block level (H_ * W_ blocks, N * F threads)");
    time_start = clock();
    conv_gpu_col(feature_map, kernel, output);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");

    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on gpu, using MatrixMultiplcation, single thread");
    time_start = clock();
    conv_gpu_matrix_single_thread(feature_map, kernel, output);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");

    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on gpu, using MatrixMultiplcation,\n "
                                 "parallel on channel level (N blocks, F threads)");
    time_start = clock();
    conv_gpu_matrix_channel(feature_map, kernel, output);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");

    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on gpu, using MatrixMultiplcation,\n "
                                 "parallel on block level (H_ * W_ blocks, N * F threads)");
    time_start = clock();
    conv_gpu_matrix_col(feature_map, kernel, output);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");

    printf("==========================================================\n");
    printf("Operation: %s\n", "2D convolution on gpu, using MatrixMultiplcation and shared memory,\n "
                "parallel N * F / TILE_WIDTH * H_ * W_ / TILE_WIDTH + 1 blocks, TILE_WIDTH  * TILE_WIDTH  threads");
    time_start = clock();
    conv_gpu_matrix_shared(feature_map, kernel, output);
    time_end = clock();
    correctness = result_is_equal(output_cpu, output);
    printf("Time use: %gms\n", 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC);
    printf("Correctness: %d\n", correctness);
    printf("==========================================================\n");


    free(feature_map); free(kernel); free(output);
    return 0;
}


//函数定义

bool result_is_equal(float * tensor1, float * tensor2) {
    for (int i = 0; i < N * F * H_ * W_; i++) {
        if (tensor1[i] != tensor2[i]) {
            float tmp = tensor1[i] - tensor2[i];
            if(tmp < 0.01 && tmp > -0.01) continue;     // 处理cpu和gpu 对float计算结果的精确度保留不一致的问题
            printf("Error point: %d, first value: %f, second value: %f\n", i, tensor1[i], tensor2[i]);
            return 0;}
    }
    return 1;
}

void conv_cpu(float * feature_map, float * kernel, float * output) {
    for (int n = 0; n < N; n++) {
        for (int f = 0; f < F; f++) {
            for (int h = 0; h < H_; h++) {
                for (int w = 0; w < W_; w++) {
                    float local_point = 0.0;
                    for (int c = 0; c < C; c++) {
                        int fm_p = n * C * H * W + c * H * W + h * W + w;
                        int kernel_p = f * C * K * K + c * K * K;
                        for (int i = 0; i < K; i++) {
                            for (int j = 0; j < K; j++) {
                                local_point += feature_map[fm_p + i * W + j] * kernel[kernel_p + i * K + j];
                            }
                        }
                    }
                    output[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] = local_point;
                }
            }
        }
    }
}

void conv_gpu_single(float *feature_map, float *kernel, float *output) {
    int size_fm = N * C * H * W;
    int size_kernel = F * C * K * K;
    int size_out = N * F * H_ * W_;
    float *feature_map_gpu, *kernel_gpu, *output_gpu;

    cudaMalloc((void **) &feature_map_gpu, size_fm * sizeof(float));
    cudaMemcpy(feature_map_gpu, feature_map, size_fm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &kernel_gpu, size_kernel * sizeof(float));
    cudaMemcpy(kernel_gpu, kernel, size_kernel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_gpu, size_out * sizeof(float));

    conv_gpu_single_kernel <<<1, 1>>> (feature_map_gpu, kernel_gpu, output_gpu);

    cudaMemcpy(output, output_gpu, size_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(feature_map_gpu); cudaFree(kernel_gpu); cudaFree(output_gpu);
}

void conv_gpu_sample(float *feature_map, float *kernel, float *output) {
    int size_fm = N * C * H * W;
    int size_kernel = F * C * K * K;
    int size_out = N * F * H_ * W_;
    float *feature_map_gpu, *kernel_gpu, *output_gpu;

    cudaMalloc((void **) &feature_map_gpu, size_fm * sizeof(float));
    cudaMemcpy(feature_map_gpu, feature_map, size_fm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &kernel_gpu, size_kernel * sizeof(float));
    cudaMemcpy(kernel_gpu, kernel, size_kernel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_gpu, size_out * sizeof(float));

    dim3 dimGrid(1);
    dim3 dimBlock(N);
    conv_gpu_sample_kernel <<<dimGrid, dimBlock>>> (feature_map_gpu, kernel_gpu, output_gpu);

    cudaMemcpy(output, output_gpu, size_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(feature_map_gpu); cudaFree(kernel_gpu); cudaFree(output_gpu);
}

__global__
void conv_gpu_sample_kernel(float *feature_map, float *kernel, float *output) {
    int n = threadIdx.x;
    
    for (int f = 0; f < F; f++) {
        // printf("%d, %d\n", n, f);
        for (int h = 0; h < H_; h++) {
            for (int w = 0; w < W_; w++) {
                float local_point = 0.0;
                for (int c = 0; c < C; c++) {
                    int fm_p = n * C * H * W + c * H * W + h * W + w;
                    int kernel_p = f * C * K * K + c * K * K;
                    for (int i = 0; i < K; i++) {
                        for (int j = 0; j < K; j++) {
                            local_point += feature_map[fm_p + i * W + j] * kernel[kernel_p + i * K + j];
                        }
                    }       
                }
                output[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] = local_point;
            }
        }
    }
}


__global__
void conv_gpu_single_kernel(float *feature_map, float *kernel, float *output) {
    for (int n = 0; n < N; n++) {
        for (int f = 0; f < F; f++) {
            for (int h = 0; h < H_; h++) {
                for (int w = 0; w < W_; w++) {
                    float local_point = 0.0;
                    for (int c = 0; c < C; c++) {
                        int fm_p = n * C * H * W + c * H * W + h * W + w;
                        int kernel_p = f * C * K * K + c * K * K;
                        for (int i = 0; i < K; i++) {
                            for (int j = 0; j < K; j++) {
                                local_point += feature_map[fm_p + i * W + j] * kernel[kernel_p + i * K + j];
                            }
                        }
                        
                    }
                    output[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] = local_point;
                }
            }
        }

    }

}

void conv_gpu_channel(float *feature_map, float *kernel, float *output) {
    int size_fm = N * C * H * W;
    int size_kernel = F * C * K * K;
    int size_out = N * F * H_ * W_;
    float *feature_map_gpu, *kernel_gpu, *output_gpu;

    cudaMalloc((void **) &feature_map_gpu, size_fm * sizeof(float));
    cudaMemcpy(feature_map_gpu, feature_map, size_fm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &kernel_gpu, size_kernel * sizeof(float));
    cudaMemcpy(kernel_gpu, kernel, size_kernel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_gpu, size_out * sizeof(float));

    dim3 dimGrid(N);
    dim3 dimBlock(F);
    conv_gpu_channel_kernel <<<dimGrid, dimBlock>>> (feature_map_gpu, kernel_gpu, output_gpu);

    cudaMemcpy(output, output_gpu, size_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(feature_map_gpu); cudaFree(kernel_gpu); cudaFree(output_gpu);
}

__global__
void conv_gpu_channel_kernel(float *feature_map, float *kernel, float *output) {
    int n = blockIdx.x;
    int f = threadIdx.x;
    for (int h = 0; h < H_; h++) {
        for (int w = 0; w < W_; w++) {
            float local_point = 0.0;
            for (int c = 0; c < C; c++) {
                int fm_p = n * C * H * W + c * H * W + h * W + w;
                int kernel_p = f * C * K * K + c * K * K;
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        local_point += feature_map[fm_p + i * W + j] * kernel[kernel_p + i * K + j];
                    }
                }       
            }
            output[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] = local_point;
        }
    }
}

void conv_gpu_col(float *feature_map, float *kernel, float *output) {
    int size_fm = N * C * H * W;
    int size_kernel = F * C * K * K;
    int size_out = N * F * H_ * W_;
    float *feature_map_gpu, *kernel_gpu, *output_gpu;

    cudaMalloc((void **) &feature_map_gpu, size_fm * sizeof(float));
    cudaMemcpy(feature_map_gpu, feature_map, size_fm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &kernel_gpu, size_kernel * sizeof(float));
    cudaMemcpy(kernel_gpu, kernel, size_kernel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_gpu, size_out * sizeof(float));

    dim3 dimGrid(H_, W_);
    dim3 dimBlock(N, F);
    conv_gpu_col_kernel <<<dimGrid, dimBlock>>> (feature_map_gpu, kernel_gpu, output_gpu);

    cudaMemcpy(output, output_gpu, size_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(feature_map_gpu); cudaFree(kernel_gpu); cudaFree(output_gpu);
}

__global__
void conv_gpu_col_kernel(float *feature_map, float *kernel, float *output) {
    int n = threadIdx.x;
    int f = threadIdx.y;
    int h = blockIdx.x;
    int w = blockIdx.y;
    float local_point = 0.0;
    for (int c = 0; c < C; c++) {
        int fm_p = n * C * H * W + c * H * W + h * W + w;
        int kernel_p = f * C * K * K + c * K * K;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                local_point += feature_map[fm_p + i * W + j] * kernel[kernel_p + i * K + j];
            }
        }       
    }
    output[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] = local_point;
}


void conv_gpu_matrix_single_thread(float *feature_map, float *kernel, float *output) {
    int size_fm_matrix = N * H_ * W_ * C * K * K;
    float * fm_matrix;
    fm_matrix = (float *) malloc(size_fm_matrix * sizeof(float));
    for (int count = 0; count < size_fm_matrix; count++) {
        int tmp = count;
        int j = tmp % K;
        tmp = tmp / K;
        int i = tmp % K;
        tmp = tmp / K;
        int c = tmp % C;
        tmp = tmp / C;
        int w = tmp % W_;
        tmp = tmp / W_;
        int h = tmp % H_;
        int n = tmp / H_;

        fm_matrix[count] =  feature_map[n * C * H * W + c * H * W + (h + i) * W + (w + j)];
    }


    int size_kernel_matrix = F * C * K * K;
    int size_out_matrix = N * F * H_ * W_;
    float *feature_map_gpu, *kernel_gpu, *output_gpu;

    cudaMalloc((void **) &feature_map_gpu, size_fm_matrix * sizeof(float));
    cudaMemcpy(feature_map_gpu, fm_matrix, size_fm_matrix * sizeof(float), cudaMemcpyHostToDevice);

    // test <<<1, 1>>> (feature_map_gpu, size_fm_matrix);
    // for (int count = 0; count < 10; count++) {
    //     if(feature_map_gpu[count]!=1)
    //     printf("%f\n", feature_map_gpu[count]);
    // }

    free(fm_matrix);

    cudaMalloc((void **) &kernel_gpu, size_kernel_matrix * sizeof(float));
    cudaMemcpy(kernel_gpu, kernel, size_kernel_matrix * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_gpu, size_out_matrix * sizeof(float));
    
    // for (int count = 0; count < 10; count++) {
    //     if(kernel_gpu[count]!=1)
    //     printf("%f\n", kernel_gpu[count]);
    // }
    dim3 dimGrid(1);
    dim3 dimBlock(1);
    conv_gpu_kernel_matrix_single_thread <<<dimGrid, dimBlock>>> (feature_map_gpu, kernel_gpu, output_gpu);

    cudaMemcpy(output, output_gpu, size_out_matrix * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(feature_map_gpu); cudaFree(kernel_gpu); cudaFree(output_gpu);
    
}

__global__
void conv_gpu_kernel_matrix_single_thread(float *feature_map_gpu, float *kernel_gpu, float *output_gpu) {
    int len = C * K * K;
    float local_point;
    // printf("==========================================");
    for (int n = 0; n < N; n++) {
        // printf("%d\n", n);
        for (int f = 0; f < F; f++) {
            // printf("%d, %d, %f\n", n, f, local_point);
            for (int h = 0; h < H_; h++) {
                for (int w = 0; w < W_; w++) {
                    local_point = 0.0;
                    for (int i = 0; i < len; i++) {
                        local_point += feature_map_gpu[n * H_ * W_ * len + h * W_ * len + w * len + i]
                                    * kernel_gpu[f * len + i];
                        // if(feature_map_gpu[n * H * W * len + h * W * len + w * len + i] != 1)
                        // printf("%d, %d, %d, %d, %d, %f\n", n, f, h, w, i, feature_map_gpu[n * H * W * len + h * W * len + w * len + i]);
                    }
                    output_gpu[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] = local_point;
                }
            }
        }
        
    }

}

void conv_gpu_matrix_channel(float *feature_map, float *kernel, float *output) {
    int size_fm_matrix = N * H_ * W_ * C * K * K;
    float * fm_matrix;
    fm_matrix = (float *) malloc(size_fm_matrix * sizeof(float));
    // im2col, [N, C, H, W] -> [N, H_ * W_, C * K * K]
    for (int count = 0; count < size_fm_matrix; count++) {
        int tmp = count;
        int j = tmp % K;
        tmp = tmp / K;
        int i = tmp % K;
        tmp = tmp / K;
        int c = tmp % C;
        tmp = tmp / C;
        int w = tmp % W_;
        tmp = tmp / W_;
        int h = tmp % H_;
        int n = tmp / H_;

        fm_matrix[count] =  feature_map[n * C * H * W + c * H * W + (h + i) * W + (w + j)];
    }


    int size_kernel_matrix = F * C * K * K;
    int size_out_matrix = N * F * H_ * W_;
    float *feature_map_gpu, *kernel_gpu, *output_gpu;

    cudaMalloc((void **) &feature_map_gpu, size_fm_matrix * sizeof(float));
    cudaMemcpy(feature_map_gpu, fm_matrix, size_fm_matrix * sizeof(float), cudaMemcpyHostToDevice);

    free(fm_matrix);

    cudaMalloc((void **) &kernel_gpu, size_kernel_matrix * sizeof(float));
    cudaMemcpy(kernel_gpu, kernel, size_kernel_matrix * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_gpu, size_out_matrix * sizeof(float));
    
    dim3 dimGrid(N);
    dim3 dimBlock(F);
    conv_gpu_kernel_matrix_channel <<<dimGrid, dimBlock>>> (feature_map_gpu, kernel_gpu, output_gpu);

    cudaMemcpy(output, output_gpu, size_out_matrix * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(feature_map_gpu); cudaFree(kernel_gpu); cudaFree(output_gpu);
    
}

__global__
void conv_gpu_kernel_matrix_channel(float *feature_map_gpu, float *kernel_gpu, float *output_gpu) {
    int len = C * K * K;
    float local_point;
    int n = blockIdx.x;
    int f = threadIdx.x;
    for (int h = 0; h < H_; h++) {
        for (int w = 0; w < W_; w++) {
            local_point = 0.0;
            for (int i = 0; i < len; i++) {
                local_point += feature_map_gpu[n * H_ * W_ * len + h * W_ * len + w * len + i]
                            * kernel_gpu[f * len + i];
            }
            output_gpu[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] = local_point;
        }
    } 
}

void conv_gpu_matrix_col(float *feature_map, float *kernel, float *output) {
    int size_fm_matrix = N * H_ * W_ * C * K * K;
    float * fm_matrix;
    fm_matrix = (float *) malloc(size_fm_matrix * sizeof(float));
    // im2col, [N, C, H, W] -> [N, H_ * W_, C * K * K]
    for (int count = 0; count < size_fm_matrix; count++) {
        int tmp = count;
        int j = tmp % K;
        tmp = tmp / K;
        int i = tmp % K;
        tmp = tmp / K;
        int c = tmp % C;
        tmp = tmp / C;
        int w = tmp % W_;
        tmp = tmp / W_;
        int h = tmp % H_;
        int n = tmp / H_;

        fm_matrix[count] =  feature_map[n * C * H * W + c * H * W + (h + i) * W + (w + j)];
    }


    int size_kernel_matrix = F * C * K * K;
    int size_out_matrix = N * F * H_ * W_;
    float *feature_map_gpu, *kernel_gpu, *output_gpu;

    cudaMalloc((void **) &feature_map_gpu, size_fm_matrix * sizeof(float));
    cudaMemcpy(feature_map_gpu, fm_matrix, size_fm_matrix * sizeof(float), cudaMemcpyHostToDevice);

    free(fm_matrix);

    cudaMalloc((void **) &kernel_gpu, size_kernel_matrix * sizeof(float));
    cudaMemcpy(kernel_gpu, kernel, size_kernel_matrix * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_gpu, size_out_matrix * sizeof(float));
    
    dim3 dimGrid(H_, W_);
    dim3 dimBlock(N, F);
    conv_gpu_kernel_matrix_col <<<dimGrid, dimBlock>>> (feature_map_gpu, kernel_gpu, output_gpu);

    cudaMemcpy(output, output_gpu, size_out_matrix * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(feature_map_gpu); cudaFree(kernel_gpu); cudaFree(output_gpu);
    
}

__global__
void conv_gpu_kernel_matrix_col(float *feature_map_gpu, float *kernel_gpu, float *output_gpu) {
    int len = C * K * K;
    float local_point;
    int n = threadIdx.x;
    int f = threadIdx.y;
    int h = blockIdx.x;
    int w = blockIdx.y;
    local_point = 0.0;
    for (int i = 0; i < len; i++) {
            local_point += feature_map_gpu[n * H_ * W_ * len + h * W_ * len + w * len + i]
                        * kernel_gpu[f * len + i];
    }
        output_gpu[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] = local_point;
}


void conv_gpu_matrix_shared(float *feature_map, float *kernel, float *output) {
    int size_fm_matrix = N * H_ * W_ * L;
    float * fm_matrix;
    
    fm_matrix = (float *) malloc(size_fm_matrix * sizeof(float));
    for (int count = 0; count < size_fm_matrix; count++) {
        int tmp = count;
        int j = tmp % K;
        tmp = tmp / K;
        int i = tmp % K;
        tmp = tmp / K;
        int c = tmp % C;
        tmp = tmp / C;
        int w = tmp % W_;
        tmp = tmp / W_;
        int h = tmp % H_;
        int n = tmp / H_;

        
        fm_matrix[count] =  feature_map[n * C * H * W + c * H * W + (h + i) * W + (w + j)];
    }

    
    int size_kernel_matrix = F * C * K * K;
    int size_out_matrix = N * F * H_ * W_;
    float *feature_map_gpu, *kernel_gpu, *output_gpu;

    cudaMalloc((void **) &feature_map_gpu, size_fm_matrix * sizeof(float));
    cudaMemcpy(feature_map_gpu, fm_matrix, size_fm_matrix * sizeof(float), cudaMemcpyHostToDevice);

    free(fm_matrix);

    cudaMalloc((void **) &kernel_gpu, size_kernel_matrix * sizeof(float));
    cudaMemcpy(kernel_gpu, kernel, size_kernel_matrix * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_gpu, size_out_matrix * sizeof(float));
    
    dim3 dimGrid(N, F / TILE_WIDTH, H_ * W_ / TILE_WIDTH + 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    conv_gpu_kernel_matrix_shared <<<dimGrid, dimBlock>>> (feature_map_gpu, kernel_gpu, output_gpu);
    
    cudaMemcpy(output, output_gpu, size_out_matrix * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(feature_map_gpu); cudaFree(kernel_gpu); cudaFree(output_gpu);
    
}

__global__
void conv_gpu_kernel_matrix_shared(float *feature_map_gpu, float *kernel_gpu, float *output_gpu) {

    __shared__ float fms[TILE_WIDTH][TILE_WIDTH];
    __shared__ float kernels[TILE_WIDTH][TILE_WIDTH];
    int n = blockIdx.x;
    int bx = blockIdx.y;
    int by = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_kernel = bx * TILE_WIDTH + tx;
    int row_fm = by * TILE_WIDTH + ty;
    int f = row_kernel;
    int h = row_fm / W_;
    int w = row_fm % W_;



    for (int a = 0; a < L/TILE_WIDTH; a++) {
        if (row_kernel < F && a * TILE_WIDTH + ty < L) {
            kernels[tx][ty] = kernel_gpu[row_kernel * L + (a * TILE_WIDTH + ty)];
        }
        else {
            kernels[tx][ty] = 0;
        }
        if (row_fm < H_ * W_ && a * TILE_WIDTH + tx < L) {
            fms[ty][tx] = feature_map_gpu[n * H_ * W_ * L + row_fm * L + (a * TILE_WIDTH + tx)];
        }
        else {
            fms[ty][tx] = 0;
        }
        __syncthreads();

        float local_point = 0.0;
        for (int i = 0; i < TILE_WIDTH; i++) {
            local_point += fms[ty][i] * kernels[tx][i];
        }
        if (n * F * H_ * W_ + f * H_ * W_ + h * W_ + w < N * F * H_ * W_)
        output_gpu[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] += local_point;

        __syncthreads();
    
    }
}