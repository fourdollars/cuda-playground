#include <iostream>
#include <fstream>

// Simple vector struct for color representation
struct Vec3 {
    float x, y, z;
};

// CUDA Kernel for Mandelbrot Set Calculation
__global__ void mandelbrot_kernel(Vec3 *pixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;
    int offset = x + y * width;

    float cx = (float)(x - 400) / 200; // Scale and translate x coordinate
    float cy = (float)(y - 300) / 200; // Scale and translate y coordinate

    float zx = 0;
    float zy = 0;

    int iter = 0;
    const int max_iter = 1000;

    while (zx * zx + zy * zy < 4.0 && iter < max_iter) {
        float temp_zx = zx * zx - zy * zy + cx;
        zy = 2 * zx * zy + cy;
        zx = temp_zx;
        iter++;
    }

    // Assign colors based on the number of iterations
    if (iter == max_iter) {
        pixels[offset] = {0.0f, 0.0f, 0.0f}; // Black for points in the set
    } else {
        float t = (float)iter / (float)max_iter;
        pixels[offset] = {0.5f * (1.0f + cos(3.14159265f * 8.0f * t)),
                          0.5f * (1.0f + cos(3.14159265f * 16.0f * t)),
                          0.5f * (1.0f + cos(3.14159265f * 24.0f * t))};
    }
}

int main() {
    const int width = 800;
    const int height = 600;

    Vec3 *d_pixels;
    cudaMalloc((void **)&d_pixels, width * height * sizeof(Vec3));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_pixels);

    Vec3 *h_pixels = (Vec3 *)malloc(width * height * sizeof(Vec3));
    cudaMemcpy(h_pixels, d_pixels, width * height * sizeof(Vec3), cudaMemcpyDeviceToHost);

    // Write the PPM image file
    std::ofstream outfile("mandelbrot.ppm");
    outfile << "P3\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Vec3 p = h_pixels[i * width + j];
            int r = (int)(p.x * 255.99);
            int g = (int)(p.y * 255.99);
            int b = (int)(p.z * 255.99);
            outfile << r << " " << g << " " << b << "\n";
        }
    }

    cudaFree(d_pixels);
    free(h_pixels);

    std::cout << "Mandelbrot set image generated as mandelbrot.ppm" << std::endl;

    return 0;
}
