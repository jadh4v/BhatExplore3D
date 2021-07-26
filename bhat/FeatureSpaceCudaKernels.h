#pragma once
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <device_atomic_functions.h>
#include <cuda_device_runtime_api.h>


// Image to Gradient
__global__ void GradientKernel(
    float4* output,
    size_t arraySize,
    cudaTextureObject_t Phi,
    const int3 texDim,
    const float3 spacing,
    const float bandwidth)
{
    // Calculate normalized texture coordinates
    int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t outIndex = idx.z*texDim.x*texDim.y + idx.y * texDim.x + idx.x;
    if (outIndex >= arraySize)
        return;

    float width = float(texDim.x);
    float height = float(texDim.y);
    float depth = float(texDim.z);
    float3 texCoord = make_float3(float(idx.x) / (width - 1.0f), float(idx.y) / (height - 1.0f), float(idx.z) / (depth - 1.0f));
    texCoord.z = texDim.z <= 1 ? 0 : texCoord.z;
    float3 delta = make_float3(1.0f / (width-1.0f), 1.0f / (height-1.0f), 1.0f / (depth-1.0f));
    delta.z = texDim.z <= 1 ? 0 : delta.z;

#if 0
    float phi_value = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z);
    if (fabs(phi_value) > (bandwidth+2)) // early return if phi update not necessary on this vertex.
    {
        output[outIndex] = make_float4(0);
        return;
    }
#endif

    float3 p_low, p_high;
    p_low.x  = tex3D<float>(Phi, texCoord.x - delta.x, texCoord.y, texCoord.z);
    p_low.y = tex3D<float>(Phi, texCoord.x, texCoord.y - delta.y, texCoord.z);
    p_low.z = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z - delta.z);

    p_high.x = tex3D<float>(Phi, texCoord.x + delta.x, texCoord.y, texCoord.z);
    p_high.y = tex3D<float>(Phi, texCoord.x, texCoord.y + delta.y, texCoord.z);
    p_high.z = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z + delta.z);

    float3 g = p_high - p_low;
    float3 denom = make_float3(1.0f / (2.0f * spacing.x), 1.0f / (2.0f*spacing.y), 1.0f / (2.0f*spacing.z));
    denom.z = texDim.z <= 1 ? 0 : denom.z;
    g = g * denom;  // element-wise multiply
    g = normalize(g);
    g.x = isnan(g.x) ? 0 : g.x;
    g.y = isnan(g.y) ? 0 : g.y;
    g.z = isnan(g.z) ? 0 : g.z;

// gradient 
    output[outIndex] = make_float4(g.x, g.y, g.z, 0.0f);
// tex coord
    //output[idx.z*texDim.x*texDim.y + idx.y * texDim.x + idx.x] = make_float4(texCoord.x, texCoord.y, texCoord.z, 0);
// p_low
    //output[idx.z*texDim.x*texDim.y + idx.y * texDim.x + idx.x] = make_float4(p_low.x, p_low.y, p_low.z, 0);
}

// Gradient to Divergence
__global__ void DivergenceKernel(
    float* output,
    size_t arraySize,
    cudaTextureObject_t Phi,
    cudaTextureObject_t Gradients,
    const int3 texDim,
    const float3 spacing,
    const float alpha,
    const float bandwidth)
{
    // Calculate normalized texture coordinates
    int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;

    size_t outIndex = idx.z*texDim.x*texDim.y + idx.y * texDim.x + idx.x;
    if (outIndex >= arraySize)
        return;

    float width = texDim.x;
    float height = texDim.y;
    float depth = texDim.z;

    float3 texCoord = make_float3(float(idx.x) / (width - 1.0f), float(idx.y) / (height - 1.0f), float(idx.z) / (depth - 1.0f));
    texCoord.z = texDim.z <= 1 ? 0 : texCoord.z;
    float3 delta = make_float3(1.0f / (width-1.0f), 1.0f / (height-1.0f), 1.0f / (depth-1.0f));
    delta.z = texDim.z <= 1 ? 0 : delta.z;
    float3 denomFactor = make_float3(1.0f / (2.0f * spacing.x), 1.0f / (2.0f*spacing.y), 1.0f / (2.0f*spacing.z));
    denomFactor.z = texDim.z <= 1 ? 0 : denomFactor.z;

#if 0
    float phi_value  = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z);
    if (fabs(phi_value) > (bandwidth+2)) // early return if phi update not necessary on this vertex.
    {
        output[outIndex] = 0;
        return;
    }
#endif

    float3 p_low, p_high;
    p_low.x = normalize(tex3D<float4>(Gradients, texCoord.x - delta.x, texCoord.y, texCoord.z)).x;
    p_low.y = normalize(tex3D<float4>(Gradients, texCoord.x, texCoord.y - delta.y, texCoord.z)).y;
    p_low.z = normalize(tex3D<float4>(Gradients, texCoord.x, texCoord.y, texCoord.z - delta.z)).z;

    p_high.x = normalize(tex3D<float4>(Gradients, texCoord.x + delta.x, texCoord.y, texCoord.z)).x;
    p_high.y = normalize(tex3D<float4>(Gradients, texCoord.x, texCoord.y + delta.y, texCoord.z)).y;
    p_high.z = normalize(tex3D<float4>(Gradients, texCoord.x, texCoord.y, texCoord.z + delta.z)).z;

    //float3 g = make_float3(p_low.x - p_high.x, p_low.y - p_high.y, p_low.z - p_high.z);
    float3 gg = p_high - p_low;
    float div = dot(gg, denomFactor);
    //float4 curr_g = tex3D<float4>(Gradients, texCoord.x, texCoord.y, texCoord.z);
    //float result = alpha*div*length(curr_g);
    float result = alpha * div;
    output[outIndex] = isnan(result) ? 0 : result;
}

template<typename T>
__device__ void filter_positive(T& value)
{
    value = (value > 0) ? value : 0;
}
template<typename T>
__device__ void filter_negative(T& value)
{
    value = (value < 0) ? value : 0;
}

template<typename T>
__device__ T sussman_sign(const T& value)
{
    return T(value / sqrtf(value*value  + 1.0));
}

__device__ void process_phi_update(signed char& updated, const float phi_value, const float result)
{
    if (result * phi_value < 0)
    {
        if (result < 0)
            updated = 1;
        else if (result > 0)
            updated = 2;
    }
    /*
    else if (fabs(phi_value) < 1e-10 && result < 0)
    {
        updated = 1;
    }
    */
}

__global__ void SussmanKernel(
                        float* output,
                        signed char* updated,
                        size_t arraySize,
                        const float dt,
                        cudaTextureObject_t Phi,
                        const int3 texDim )
{
    // Calculate normalized texture coordinates
    int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t outIndex = idx.z*texDim.x*texDim.y + idx.y * texDim.x + idx.x;
    if (outIndex >= arraySize)
        return;

    float width = float(texDim.x);
    float height = float(texDim.y);
    float depth = float(texDim.z);
    float3 texCoord = make_float3(float(idx.x) / (width - 1.0f), float(idx.y) / (height - 1.0f), float(idx.z) / (depth - 1.0f));
    texCoord.z = texDim.z <= 1 ? 0 : texCoord.z;
    float3 delta = make_float3(1.0f / (width-1.0f), 1.0f / (height-1.0f), 1.0f / (depth-1.0f));
    delta.z = texDim.z <= 1 ? 0 : delta.z;

    float dD = 0.0f;

    // collect neighborhood samples
    float phi_value = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z);
    float xm = tex3D<float>(Phi, texCoord.x - delta.x, texCoord.y, texCoord.z);
    float xp = tex3D<float>(Phi, texCoord.x + delta.x, texCoord.y, texCoord.z);
    float ym = tex3D<float>(Phi, texCoord.x, texCoord.y - delta.y, texCoord.z);
    float yp = tex3D<float>(Phi, texCoord.x, texCoord.y + delta.y, texCoord.z);
    float zm = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z - delta.z);
    float zp = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z + delta.z);

    // compute differences
    xm = phi_value - xm;
    xp = xp - phi_value;
    ym = phi_value - ym;
    yp = yp - phi_value;
    zm = phi_value - zm;
    zp = zp - phi_value;

    if (phi_value > 0)
    {
        filter_positive(xm);  filter_negative(xp);
        filter_positive(ym);  filter_negative(yp);
        filter_positive(zm); filter_negative(zp);
    }
    else
    {
        filter_negative(xm);  filter_positive(xp);
        filter_negative(ym);  filter_positive(yp);
        filter_negative(zm); filter_positive(zp);
    }

    // Convert to 2D equation if Z-dimension is 1 or less.
    // Everything else should be common to both 2D and 3D cases.
    zm = texDim.z <= 1 ? 0 : zm;
    zp = texDim.z <= 1 ? 0 : zp;

    dD = sqrtf(fmaxf(xm*xm, xp*xp) + fmaxf(ym*ym, yp*yp) + fmaxf(zm*zm, zp*zp)) - 1.0f;
    //output[outIndex] = phi_value - dt * sussman_sign(phi_value) * dD;
    float result = phi_value - dt * sussman_sign(phi_value) * dD;
    if (!isnan(result))
    {
        output[outIndex] = result;
        process_phi_update(updated[outIndex], phi_value, result);
    }
}

// Update Phi, Process Divergence
__global__ void UpdatePhiKernel(float* phi, size_t arraySize, const float* ak, signed char* updated, const char* blanked, const float3 params)
{
    const float eps = float(1e-16);
    const float narrowband = params.x;
    const float stepSize = params.y;
    const float max_e = params.z;

    size_t outIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (outIndex >= arraySize)// || blanked[outIndex] == false)
        return;

    if (blanked[outIndex] == 1)
    {
        phi[outIndex] = 5.0f;
        return;
    }

    float phi_value = phi[outIndex];
    if (fabs(phi_value) <= narrowband)
    {
        float result = phi_value + stepSize / (max_e + eps) * ak[outIndex];
        if (!isnan(result))
        {
            phi[outIndex] = result;
            process_phi_update(updated[outIndex], phi_value, result);
        }
    }
}

__global__ void Histogram2D_kernel(const float2* points, const size_t arraySize, float* H, size_t histDimSize)
{
    // Calculate thread-id
    int gId = blockIdx.x * blockDim.x + threadIdx.x;
    if (gId >= arraySize)
        return;

    float2 attrib = points[gId];

    dim3 index;
    index.x = int(attrib.x);
    index.y = int(attrib.y);
    int location = index.y*histDimSize + index.x;
    if(location < histDimSize*histDimSize)
        atomicAdd(&(H[location]), 1.0f);
}

__global__ void Histogram3D_kernel(const float3* points, const size_t arraySize, float* H, size_t histDimSize)
{
    // Calculate thread-id
    int gId = blockIdx.x * blockDim.x + threadIdx.x;
    if (gId >= arraySize)
        return;

    float3 attrib = points[gId];

    int3 index;
    index.x = int(attrib.x);
    index.y = int(attrib.y);
    index.z = int(attrib.z);
    int location = index.z*(histDimSize*histDimSize) + index.y*histDimSize + index.x;
    if(location < histDimSize*histDimSize*histDimSize)
        atomicAdd(&(H[location]), 1.0f);
}

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
__constant__ float c_Kernel[KERNEL_LENGTH];
__global__ void ConvolveKernel(float* output, size_t arraySize, const float3 mode, cudaTextureObject_t texObj, const int3 texDim)
{
    int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;

    size_t outIndex = idx.z*texDim.x*texDim.y + idx.y * texDim.x + idx.x;
    if (outIndex >= arraySize)
        return;

    const float3 anchor = make_float3( (float)idx.x + 0.5f,
                                       (float)idx.y + 0.5f,
                                       (float)idx.z + 0.5f);

    float sum = 0.0f;

//#pragma unroll
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        //float3 offset = float(k) * mode;
        float3 offset = make_float3(float(k),float(k),float(k)) * mode;
        float3 pos = anchor + offset;
        float k_value = c_Kernel[KERNEL_RADIUS + k];
        float texValue = tex3D<float>(texObj, pos.x, pos.y, pos.z);
        sum += texValue * k_value;
    }

    output[outIndex] = sum;
}

/*
template<typename T>
__global__ void UpdateHistograms(float* Hin, float* Hout, const signed char* updated, const T* attribs, const size_t arraySize, const size_t hsz)
{
    size_t gId = blockIdx.x * blockDim.x + threadIdx.x;
    if (gId >= arraySize)
        return;

    int updateFlag = int(updated[gId]);
    if (updateFlag == 0)
        return;

    T att = attribs[gId];
    size_t outIdx = attrib_to_index(att, hsz);
    // function to avoid if condition.
    float value = -2.0f*updateFlag + 3.0f; 
    atomicAdd(&Hin[outIdx],  value);
    atomicAdd(&Hout[outIdx], -1.0f*value);
}
*/

__global__ void UpdateHistograms(float* Hin, float* Hout, const signed char* updated, const float* attribs, const char* blanked, const size_t arraySize, const size_t hsz)
{
    size_t gId = blockIdx.x * blockDim.x + threadIdx.x;
    if (gId >= arraySize)
        return;

    if (blanked[gId] != 0)
        return;

    int updateFlag = int(updated[gId]);
    if (updateFlag == 0)
        return;

    size_t bucket = size_t(nearbyintf(attribs[gId]));
    if (bucket >= hsz)
        return;

    // function to avoid if condition.
    float value = -2.0f*updateFlag + 3.0f; 
    atomicAdd(&Hin[bucket],  value);
    atomicAdd(&Hout[bucket], -1.0f*value);
}

__global__ void UpdateHistograms(float* Hin, float* Hout, const signed char* updated, const float2* attribs, const char* blanked, const size_t arraySize, const size_t hsz)
{
    size_t gId = blockIdx.x * blockDim.x + threadIdx.x;
    if (gId >= arraySize)
        return;

    if (blanked[gId] != 0)
        return;

    int updateFlag = int(updated[gId]);
    if (updateFlag == 0)
        return;

    float2 att = attribs[gId];
    int2 bucket = make_int2(nearbyintf(att.x), nearbyintf(att.y));
    if (bucket.x >= hsz || bucket.y >= hsz)
        return;

    int outIdx = bucket.x + hsz * bucket.y;
    // function to avoid if condition.
    float value = -2.0f*updateFlag + 3.0f; 

    atomicAdd(&Hin[outIdx],  value);
    atomicAdd(&Hout[outIdx], -1.0f*value);
}

__global__ void UpdateHistograms(float* Hin, float* Hout, const signed char* updated, const float3* attribs, const char* blanked, const size_t arraySize, const size_t hsz)
{
    size_t gId = blockIdx.x * blockDim.x + threadIdx.x;
    if (gId >= arraySize)
        return;

    if (blanked[gId] != 0)
        return;

    int updateFlag = int(updated[gId]);
    if (updateFlag == 0)
        return;

    float3 att = attribs[gId];
    int3 bucket = make_int3(nearbyintf(att.x), nearbyintf(att.y), nearbyintf(att.z) );
    if (bucket.x >= hsz || bucket.y >= hsz || bucket.z >= hsz)
        return;

    int outIdx = bucket.x + hsz*bucket.y + hsz*hsz*bucket.z;
    // function to avoid if condition.
    float value = -2.0f*updateFlag + 3.0f; 

    atomicAdd(&Hin[outIdx],  value);
    atomicAdd(&Hout[outIdx], -1.0f*value);
}

__global__ void UpdateHistograms2(float* gHin, float* gHou, const signed char* gUpdated, const float* gAttribs, const size_t cArraySize, const size_t cHsz)
{
    __shared__ float s_Hin[256];
    __shared__ float s_Hou[256];
    size_t gId = blockIdx.x * blockDim.x + threadIdx.x;
    if (gId >= cArraySize)
        return;

    int updateFlag = int(gUpdated[gId]);
    if (updateFlag == 0)
        return;

    // initialize shared memory
    if (threadIdx.x < cHsz)
    {
        s_Hin[threadIdx.x] = 0.0f;
        s_Hou[threadIdx.x] = 0.0f;
    }

    // finish all initialization.
    __syncthreads();

    size_t bucket = size_t(gAttribs[gId]);
    if (bucket >= cHsz)
        return;

    // function to avoid if condition.
    float value = -2.0f*updateFlag + 3.0f;
    atomicAdd(&s_Hin[bucket],       value);
    atomicAdd(&s_Hou[bucket], -1.0f*value);

    // finish local updates.
    __syncthreads();

    // update the global histograms
    atomicAdd(&gHin[threadIdx.x], s_Hin[threadIdx.x]);
    atomicAdd(&gHou[threadIdx.x], s_Hou[threadIdx.x]);
    //atomicAdd(&gHin[bucket],  value);
    //atomicAdd(&gHou[bucket], -1.0f*value);
}

__device__ inline void zero_if_nan(float3& value)
{
    value.x = isnan(value.x) ? 0 : value.x;
    value.y = isnan(value.y) ? 0 : value.y;
    value.z = isnan(value.z) ? 0 : value.z;
}
__device__ float3 Gradient(
    cudaTextureObject_t Phi,
    float3 texCoord,
    float3 delta)
{
    float3 p_low, p_high;
    p_low.x = tex3D<float>(Phi, texCoord.x - delta.x, texCoord.y, texCoord.z);
    p_low.y = tex3D<float>(Phi, texCoord.x, texCoord.y - delta.y, texCoord.z);
    p_low.z = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z - delta.z);

    p_high.x = tex3D<float>(Phi, texCoord.x + delta.x, texCoord.y, texCoord.z);
    p_high.y = tex3D<float>(Phi, texCoord.x, texCoord.y + delta.y, texCoord.z);
    p_high.z = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z + delta.z);

    return (p_high - p_low);
}

// Image to Gradient
__global__ void DirectDivergenceKernel(
    float* output_V,
    size_t arraySize,
    cudaTextureObject_t Phi,
    const int3 texDim,
    const float3 spacing,
    const float bandwidth,
    const float alpha)
{
    // Calculate normalized texture coordinates
    int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t outIndex = idx.z*texDim.x*texDim.y + idx.y * texDim.x + idx.x;
    if (outIndex >= arraySize)
        return;

    float width = float(texDim.x);
    float height = float(texDim.y);
    float depth = float(texDim.z);
    float3 texCoord = make_float3(float(idx.x) / (width - 1.0f), float(idx.y) / (height - 1.0f), float(idx.z) / (depth - 1.0f));
    texCoord.z = texDim.z <= 1 ? 0 : texCoord.z;
    float3 delta = make_float3(1.0f / (width-1.0f), 1.0f / (height-1.0f), 1.0f / (depth-1.0f));
    delta.z = texDim.z <= 1 ? 0 : delta.z;

#if 1
    float phi_value = tex3D<float>(Phi, texCoord.x, texCoord.y, texCoord.z);
    if (fabs(phi_value) > (bandwidth)) // early return if phi update not necessary on this vertex.
        return;
#endif

    //float3 denom = make_float3(1.0f / spacing.x, 1.0f / spacing.y, 1.0f / spacing.z);
    float3 denom = make_float3(1.0f / (2.0f * spacing.x), 1.0f / (2.0f*spacing.y), 1.0f / (2.0f*spacing.z));
    denom.z = texDim.z <= 1 ? 0 : denom.z;

    float3 mx = Gradient(Phi, make_float3(texCoord.x - delta.x, texCoord.y, texCoord.z), delta);
    float3 px = Gradient(Phi, make_float3(texCoord.x + delta.x, texCoord.y, texCoord.z), delta);
    float3 my = Gradient(Phi, make_float3(texCoord.x, texCoord.y - delta.x, texCoord.z), delta);
    float3 py = Gradient(Phi, make_float3(texCoord.x, texCoord.y + delta.x, texCoord.z), delta);
    float3 mz = Gradient(Phi, make_float3(texCoord.x, texCoord.y, texCoord.z - delta.z), delta);
    float3 pz = Gradient(Phi, make_float3(texCoord.x, texCoord.y, texCoord.z + delta.z), delta);

    mx = normalize(mx*denom);
    px = normalize(px*denom);
    my = normalize(my*denom);
    py = normalize(py*denom);
    mz = normalize(mz*denom);
    pz = normalize(pz*denom);

    zero_if_nan(mx);
    zero_if_nan(px);
    zero_if_nan(my);
    zero_if_nan(py);
    zero_if_nan(mz);
    zero_if_nan(pz);

    float3 gg = make_float3(px.x - mx.x, py.y - my.y, pz.z - mz.z);
    //float3 gg = p_high - p_low;
    float div = dot(gg, denom);
    float result = alpha * div;
    result = isnan(result) ? 0 : result;
    float V = output_V[outIndex];
    output_V[outIndex] = result - copysignf(0.25f,V);
    //output_V[outIndex] = result - copysignf(0.5f,V);
    //output_V[outIndex] = result - copysignf(1.0f,V);
    //output_V[outIndex] = result - output_V[outIndex];
}

// Designed for small sized vectors, assumes single block.
__global__ void SumKernel(float* buffer, const float* input, size_t arraySize)
{
	__shared__ float out[512];

	size_t gridStride = blockDim.x * gridDim.x;
	size_t outIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// Run block level summation
	for (size_t i=outIndex; i < arraySize; i += gridStride)
	{
		out[i] = input[i];
		__syncthreads();
		int factor = 2;
		while (1)
		{
			int secondIndex = i + (factor/2);
			if (i % factor != 0 || secondIndex >= arraySize)
				break;

			out[i] = out[i] + out[secondIndex];
			factor *= 2;
			__syncthreads();
		}

		if (threadIdx.x == 0)
			atomicAdd(&(buffer[0]), out[blockIdx.x]);

		//if (threadIdx.x == 0)
			//buffer[blockIdx.x] = out[0];
	}

	/*
	// Add block results
	for (size_t i = outIndex; i < blockDim.x; i += gridStride)
	{
		int factor = 2;
		while (1)
		{
			int secondIndex = i + factor;
			if (i % factor != 0 || secondIndex >= arraySize)
				break;

			out[i] = out[i] + out[secondIndex];
			factor *= 2;
			__syncthreads();
		}

		if (threadIdx.x == 0)
			buffer[blockIdx.x] = out[0];
	}
	*/
}
