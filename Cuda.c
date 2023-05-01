#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "pgmio.h"

#define W 256
#define H 256

#define BlockW 8
#define BlockH 8

float inputimage[H][W];
float outputimage[H][W];

void load_image();
void call_kernel();
void save_image();

void call_kernel() {
    float *datainput, *dataoutput;
    cudaMalloc((void **)&datainput, W * H * sizeof(float));
    cudaMalloc((void **)&dataoutput, W * H * sizeof(float));

    cudaMemcpy(datainput, inputimage, W * H * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BlockW, BlockH);
    dim3 numBlocks(ceil((float)W / BlockW), ceil((float)H / BlockH));

    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total);

    imageBlur <<<numBlocks, threadsPerBlock>>> (datainput, dataoutput, W, H);

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    float total_kernel_time = 0;
    cudaEventElapsedTime(&total_kernel_time, start_total, stop_total);
    printf("Total kernel time: %f ms\n", total_kernel_time);

    cudaMemcpy(outputimage, dataoutput, W * H * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(datainput);
    cudaFree(dataoutput);
}

void save_image() {
    pgmwrite("output.pgm", outputimage, W, H);
}

int main() {
    load_image();
    call_kernel();
    save_image();
    return 0;
}
