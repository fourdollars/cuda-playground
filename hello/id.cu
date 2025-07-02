#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// CUDA kernel to calculate and print a unique global thread ID for each thread.
__global__ void hello_cuda_with_id(){
    // Calculate the unique ID for the thread within its block.
    int threadId_in_block = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    // Calculate the unique ID for the block within the grid.
    int blockId_in_grid = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;

    // Calculate the total number of threads in a block.
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    // Calculate the global thread ID.
    int globalThreadId = blockId_in_grid * threads_per_block + threadId_in_block;

    printf("Hello from global thread ID: %d\n", globalThreadId);
}

int main(){
    // Define the dimensions of the grid and blocks.
    // This creates a 3x5x7 grid of blocks, with each block containing 2x4x8 threads.
    dim3 dimGrid(3, 5, 7);
    dim3 dimBlock(2, 4, 8);

    // Launch the hello_cuda_with_id kernel with the specified grid and block dimensions.
    hello_cuda_with_id<<<dimGrid, dimBlock>>>();

    // Block until the device has completed all preceding requested tasks.
    cudaDeviceSynchronize();

    // Destroy all allocations and reset all state on the current device.
    cudaDeviceReset();

    return 0;
}
