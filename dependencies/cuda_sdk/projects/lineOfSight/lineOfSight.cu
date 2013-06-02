/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

// This sample is an implementation of a simple line-of-sight algorithm:
// Given a height map and a ray originating at some observation point,
// it computes all the points along the ray that are visible from the
// observation point.
// It is based on the description made in "Guy E. Blelloch. Vector models
// for data-parallel computing. MIT Press, 1990" and uses the parallel scan
// primitive provided by the CUDPP library (http://www.gpgpu.org/developer/cudpp/).

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, library
#include "cudpp/cudpp.h"

// includes, project
#include "cutil.h"
#include "cutil_math.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, types

// Boolean
typedef unsigned char Bool;
enum {
    False = 0,
    True = 1
};

// 2D height field
struct HeightField {
	int     width;
    float*  height;
};

// Ray
struct Ray {
	float3 origin;
    float2 dir;
    int    length;
    float  oneOverLength;
};

////////////////////////////////////////////////////////////////////////////////
// declaration, variables

// Height field texture reference
texture<float, 2, cudaReadModeElementType> g_HeightFieldTex;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
__global__ void computeAngles_kernel(const Ray, float*);
__global__ void computeVisibilities_kernel(const float*, const float*, int, Bool*);
void lineOfSight_gold(const HeightField, const Ray, Bool*);
__device__ __host__ float2 getLocation(const Ray, int);
__device__ __host__ float getAngle(const Ray, float2, float);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a line-of-sight test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv) 
{
    ////////////////////////////////////////////////////////////////////////////
    // Device initialization
    CUT_DEVICE_INIT(argc, argv);
    
    ////////////////////////////////////////////////////////////////////////////
    // Timer
    
    // Create
    uint timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    
    // Number of iterations to get accurate timing
#ifdef __DEVICE_EMULATION__
    uint numIterations = 1;
#else
    uint numIterations = 100;
#endif

    ////////////////////////////////////////////////////////////////////////////
    // Height field
    
    HeightField heightField;
    
    // Allocate in host memory
    int2 dim = make_int2(10000, 100);
    heightField.width = dim.x;
    int heightFieldSize = dim.x * dim.y * sizeof(float);
    CUT_SAFE_MALLOC(heightField.height = (float*)malloc(heightFieldSize));
    
    // Fill in with an arbitrary sine surface
    for (int x = 0; x < dim.x; ++x)
        for (int y = 0; y < dim.y; ++y) {
            float amp = 0.1f * (x + y);
            float period = 2.0f + amp;
            *(heightField.height + dim.x * y + x) =
                amp * (sinf(sqrtf((float)(x * x + y * y)) * 2.0f * 3.1416f / period) + 1.0f);
        }
    
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* heightFieldArray;
    CUDA_SAFE_CALL(cudaMallocArray(&heightFieldArray, &channelDesc, dim.x, dim.y));
     
    // Initialize device memory
    CUDA_SAFE_CALL(cudaMemcpyToArray(heightFieldArray, 0, 0, heightField.height,
                                     heightFieldSize, cudaMemcpyHostToDevice));

    // Set texture parameters
    g_HeightFieldTex.addressMode[0] = cudaAddressModeClamp;
    g_HeightFieldTex.addressMode[1] = cudaAddressModeClamp;
    g_HeightFieldTex.filterMode = cudaFilterModePoint;
    g_HeightFieldTex.normalized = 0;

    // Bind CUDA array to texture reference
    CUDA_SAFE_CALL(cudaBindTextureToArray(g_HeightFieldTex, heightFieldArray,
                                          channelDesc));
                                          
    ////////////////////////////////////////////////////////////////////////////
    // Ray (starts at origin and traverses the height field diagonally)
    
    Ray ray;
    ray.origin = make_float3(0, 0, 2.0f);
    int2 dir = make_int2(dim.x - 1, dim.y - 1); 
    ray.dir = make_float2((float)dir.x, (float)dir.y);
    ray.length = max(abs(dir.x), abs(dir.y));
    ray.oneOverLength = 1.0f / ray.length;

    ////////////////////////////////////////////////////////////////////////////
    // View angles

    // Allocate view angles for each point along the ray
    float* d_angles;
    int raySize = ray.length * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_angles, raySize));
    
    // Allocate result of max-scan operation on the array of view angles
    float* d_scannedAngles;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_scannedAngles, raySize));
    
    ////////////////////////////////////////////////////////////////////////////
    // Visibility results

    // Allocate visibility results for each point along the ray
    Bool* d_visibilities;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_visibilities, raySize));
    Bool* h_visibilities;
    CUT_SAFE_MALLOC(h_visibilities = (Bool*)malloc(raySize));
    Bool* h_visibilitiesRef;
    CUT_SAFE_MALLOC(h_visibilitiesRef = (Bool*)malloc(raySize));
    
    ////////////////////////////////////////////////////////////////////////////
    // Reference solution
    lineOfSight_gold(heightField, ray, h_visibilitiesRef);
    
    ////////////////////////////////////////////////////////////////////////////
    // Device solution

    // Execution configuration
    dim3 block(256);
    dim3 grid((uint)ceil(ray.length / (double)block.x));
    
    // Scan configuration
    CUDPPHandle scanPlan;
   
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SCAN;
    config.op = CUDPP_MAX;
    config.datatype = CUDPP_FLOAT;  
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    cudppPlan(&scanPlan, config, ray.length, 1, 0);
    
    // Compute device solution
    printf("Line of sight\n");
    cutStartTimer(timer);
    for (uint i = 0; i < numIterations; ++i) {
    
        // Compute view angle for each point along the ray
        computeAngles_kernel<<<grid, block>>>(ray, d_angles);
        CUT_CHECK_ERROR("Kernel execution failed");
        
        // Perform a max-scan operation on the array of view angles
        cudppScan(scanPlan, d_scannedAngles, d_angles, ray.length);
        CUT_CHECK_ERROR("Kernel execution failed");
        
        // Compute visibility results based on the array of view angles
        // and its scanned version
        computeVisibilities_kernel<<<grid, block>>>(d_angles, d_scannedAngles,
                                                    ray.length, d_visibilities);
        CUT_CHECK_ERROR("Kernel execution failed");
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    cudppDestroyPlan(scanPlan);
    CUT_CHECK_ERROR("Kernel execution failed");

    // Copy visibility results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_visibilities, d_visibilities, raySize,
                              cudaMemcpyDeviceToHost));
                              
    // Compare device visibility results against reference results
    CUTBoolean res = cutCompareub(h_visibilitiesRef, h_visibilities, ray.length);
    printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");
    printf("Average time: %f ms\n\n", cutGetTimerValue(timer) / numIterations);
    cutResetTimer(timer);

    // Cleanup memory
    free(heightField.height);
    free(h_visibilities);
    free(h_visibilitiesRef);
    CUDA_SAFE_CALL(cudaFree(d_angles));
    CUDA_SAFE_CALL(cudaFree(d_scannedAngles));
    CUDA_SAFE_CALL(cudaFree(d_visibilities));
    CUDA_SAFE_CALL(cudaFreeArray(heightFieldArray));
}

////////////////////////////////////////////////////////////////////////////////
//! Compute view angles for each point along the ray
//! @param ray         ray
//! @param angles      view angles
////////////////////////////////////////////////////////////////////////////////
__global__ void computeAngles_kernel(const Ray ray, float* angles)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < ray.length) {
        float2 location = getLocation(ray, i + 1);
        float height = tex2D(g_HeightFieldTex, location.x, location.y);
        float angle = getAngle(ray, location, height);
        angles[i] = angle;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute visibility for each point along the ray
//! @param angles          view angles
//! @param scannedAngles   max-scanned view angles
//! @param numAngles       number of view angles
//! @param visibilities    boolean array indicating the visibility of each point
//!                        along the ray
////////////////////////////////////////////////////////////////////////////////
__global__ void computeVisibilities_kernel(const float* angles,
                                           const float* scannedAngles,
                                           int numAngles,
                                           Bool* visibilities)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numAngles)
        visibilities[i] = scannedAngles[i] <= angles[i];
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! @param heightField     height field
//! @param ray             ray
//! @param visibilities    boolean array indicating the visibility of each point
//!                        along the ray
////////////////////////////////////////////////////////////////////////////////
void lineOfSight_gold(const HeightField heightField, const Ray ray,
                      Bool* visibilities)
{
    float angleMax = asinf(-1.0f);
    for (int i = 0; i < ray.length; ++i) {
        float2 location = getLocation(ray, i + 1);
        float height = *(heightField.height
                       + heightField.width * (int)floorf(location.y)
                       + (int)floorf(location.x));
        float angle = getAngle(ray, location, height);
        if (angle > angleMax) {
            angleMax = angle;
            visibilities[i] = True;
        }
        else
            visibilities[i] = False;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute the 2D coordinates of the point located at i steps from the origin
//! of the ray
//! @param ray      ray
//! @param i        integer offset along the ray
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ float2 getLocation(const Ray ray, int i)
{
    float step = i * ray.oneOverLength;
    return make_float2(ray.origin.x, ray.origin.y) + step * ray.dir;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute the angle of view between a 3D point and the origin of the ray
//! @param ray        ray
//! @param location   2D coordinates of the input point
//! @param height     height of the input point
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ float getAngle(const Ray ray, float2 location, float height)
{
    float2 dir = location - make_float2(ray.origin.x, ray.origin.y);
    return atanf((height - ray.origin.z) / length(dir));
}
