cmake_minimum_required(VERSION 3.28)

project("hello" LANGUAGES CXX CUDA)

add_executable(hello)
target_sources(hello PRIVATE "hello.cu")

add_executable(id)
target_sources(id PRIVATE "id.cu")
