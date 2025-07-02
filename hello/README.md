# CUDA Hello World

This is a simple "Hello World" program written in CUDA.

## Prerequisites

*   A CUDA-enabled GPU.
*   The NVIDIA CUDA Toolkit.

## Building and Running

1.  **Configure the project with CMake.**

    You need to tell CMake which CUDA architecture to target. You can find this value by running the following command:

    ```bash
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    ```

    This will output a version number like `8.6`. You should use the integer representation of this number (e.g., `86`) for the `CMAKE_CUDA_ARCHITECTURES` variable.

    If the CUDA compiler (`nvcc`) is not in your system's `PATH`, you will also need to specify its location. You can find it using:
    ```bash
    find /usr/local /opt -name nvcc
    ```

    Now, run CMake with the correct architecture and compiler path. Replace `86` with your architecture and `/usr/local/cuda-12.8/bin/nvcc` with the path to your `nvcc`.

    ```bash
    cmake . -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
    ```

2.  **Build the project.**

    Once CMake has successfully configured the project, you can build it using `make`:

    ```bash
    make
    ```

3.  **Run the executable.**

    After the build is complete, you can run the program:

    ```bash
    ./main
    ```

    You should see the output: `Hello CUDA world`
