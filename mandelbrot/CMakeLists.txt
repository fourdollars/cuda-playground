cmake_minimum_required(VERSION 3.28)

project("mandelbrot" LANGUAGES CXX CUDA)

add_executable(mandelbrot_cuda mandelbrot.cu)

add_executable(mandelbrot_cpu mandelbrot.cc)
target_compile_options(mandelbrot_cpu PRIVATE -O3)
