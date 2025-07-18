#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>

// Simple vector struct for color representation
struct Vec3 {
    float x, y, z;
};

void mandelbrot_cpu(Vec3 *pixels, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
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
                pixels[offset] = Vec3{0.0f, 0.0f, 0.0f}; // Black for points in the set
            } else {
                float t = (float)iter / (float)max_iter;
                pixels[offset] = Vec3{0.5f * (1.0f + cos(3.14159265f * 8.0f * t)),
                                  0.5f * (1.0f + cos(3.14159265f * 16.0f * t)),
                                  0.5f * (1.0f + cos(3.14159265f * 24.0f * t))};
            }
        }
    }
}

int main() {
    const int width = 800;
    const int height = 600;

    std::vector<Vec3> h_pixels(width * height);

    auto start = std::chrono::high_resolution_clock::now();
    mandelbrot_cpu(h_pixels.data(), width, height);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "CPU execution time: " << diff.count() << " s\n";

    // Write the PPM image file
    std::ofstream outfile("mandelbrot_cpu.ppm");
    start = std::chrono::high_resolution_clock::now();
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
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Image output time: " << diff.count() << " s\n";

    std::cout << "Mandelbrot set image generated as mandelbrot_cpu.ppm" << std::endl;

    return 0;
}
