#include <cstdio>
#include <vector>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
CUDAKernel_DistanceFunction(const float *A, const float *B, float *C, const int numElements_A, const int numElements_B)
{
    constexpr int threadCnt = 1024;
    __shared__ float blockDist[threadCnt];
    const int localThreadId = threadIdx.x;
    int pos = blockIdx.x * 3; 
    if (pos < numElements_A)
    {
        float4 vec_A = make_float4(A[pos], A[pos + 1], A[pos + 2], 0);
        float4 diff;
        float dist = 1000000000.0f;
        int iterationsPerThread = (numElements_B / 3 + threadCnt - 1) / threadCnt;
        //iterationsPerThread += (numElements_B / 3) % threadCnt > 0 ? 1 : 0;
        for (int k = 0; k < iterationsPerThread; ++k)
        {
            int B_pointId = (threadCnt * k + localThreadId) * 3;
            if (B_pointId < numElements_B)
            {
                float4 vec_B = make_float4(B[B_pointId], B[B_pointId + 1], B[B_pointId + 2], 0);
                diff = vec_A - vec_B;
                float d = length(diff);
                dist = fminf(dist, d);
            }
        }
        blockDist[localThreadId] = dist;
        __syncthreads();

#if 0
        // do simple iterative min find on local values
        dist = 1000000000.0f;
        if (localThreadId == 0)
        {
            for (int i = 0; i < threadCnt; ++i)
                dist = fminf(dist, blockDist[i]);

            C[blockIdx.x] = dist;
        }
#else
        // Perform interleaved-hierarchical search for minimum dist value
        int n = 2;
        for (int i = 0; i < 10/*log_2(threadCnt)*/; ++i)
        {
            if (localThreadId % n == 0)
            {
                blockDist[localThreadId] = fminf(blockDist[localThreadId], blockDist[localThreadId + n / 2]);
            }
            n *= 2;
            __syncthreads();
        }
        if (localThreadId == 0)
            C[blockIdx.x] = blockDist[0];
#endif
    }
}
__global__ void
CUDAKernel_SimpleDistanceFunction(const float *A, const float *B, float *C, const int numElements_A, const int numElements_B)
{
    const int globalThreadId = blockDim.x * blockIdx.x + threadIdx.x;
    int pos = globalThreadId * 3; 
    if (pos < numElements_A)
    {
        float4 vec_A = make_float4(A[pos], A[pos + 1], A[pos + 2], 0);
        float4 diff;
        float dist = 1000000000.0f;
        const int B_point_cnt = numElements_B / 3;
        for (int k = 0; k < B_point_cnt; ++k)
        {
            float4 vec_B = make_float4(B[k * 3], B[k * 3 + 1], B[k * 3 + 2], 0);
            diff = vec_A - vec_B;
            float d = length(diff);
            dist = fminf(dist, d);
        }
        C[blockIdx.x] = dist;
    }
}
 
int CUDAWrapper_ComputeDistanceField(const std::vector<float>& h_A, const std::vector<float>& h_B, std::vector<float>& h_C)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int numElements_A = (int)h_A.size();
    int numElements_B = (int)h_B.size();
    size_t size_A = numElements_A * sizeof(float);
    size_t size_B = numElements_B * sizeof(float);
    size_t size_C = size_A / 3;

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
#if 0
    int threadsPerBlock = 1;
    int blocksPerGrid =(numElements_A/3 + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    CUDAKernel_SimpleDistanceFunction<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements_A, numElements_B);
#else
    int threadsPerBlock = 1024;
    //int blocksPerGrid =(numElements_A/3 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = (numElements_A / 3);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    CUDAKernel_DistanceFunction<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements_A, numElements_B);
#endif
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch DistanceFunction kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    //fprintf(stderr, "size_C = %llu\n", size_C);
    err = cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        fprintf(stderr, "size_C = %llu\n", size_C);
        exit(EXIT_FAILURE);
    }

    /*
    // Verify that the result vector is correct
    for (int n = 0; n < numElements_A/3; ++n)
    {
        float diff[3];
        float dist = 1000000.0f;
        for (int k = 0; k < numElements_B / 3; ++k)
        {
            for(int i=0;i<3;++i)
                diff[i] = h_A[n*3+i] - h_B[k*3+i];

            float d = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
            dist = dist < d? dist : d;
        }
        //fprintf(stderr, "h_C[%d] = %f  !=  dist = %f\n", n, h_C[n], dist );
        if( fabs(h_C[n] - dist) > 1e-3)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", n);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");
    */

    // Free device global memory
    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_C);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

