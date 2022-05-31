#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>


template<typename T>
class CudaPtr {
public:
    CudaPtr(T *bind, int n = 1) : bind(bind), bind_cuda(nullptr), n(n) {
        cudaMalloc(&bind_cuda, n * sizeof(T));
        cudaMemcpy(bind_cuda, bind, n * sizeof(T), cudaMemcpyHostToDevice);
    }

    CudaPtr(const CudaPtr<T> &) = delete;

    ~CudaPtr() {
        cudaMemcpy(bind, bind_cuda, n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(bind_cuda);
    }

    T* operator()() {
        return bind_cuda;
    }

private:
    int n;
    T *bind;
    T *bind_cuda;
};
