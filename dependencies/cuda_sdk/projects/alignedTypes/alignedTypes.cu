/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This is a simple test showing huge access speed gap
 * between aligned and misaligned structures
 * (those having/missing __align__ keyword).
 * It measures per-element copy throughput for 
 * aligned and misaligned structures on 
 * big chunks of data.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>



////////////////////////////////////////////////////////////////////////////////
// Misaligned types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned char uint8;

typedef unsigned short int uint16;

typedef struct{
    unsigned char r, g, b, a;
} RGBA8_misaligned;

typedef struct{
    unsigned int l, a;
} LA32_misaligned;

typedef struct{
    unsigned int r, g, b;
} RGB32_misaligned;

typedef struct{
    unsigned int r, g, b, a;
} RGBA32_misaligned;



////////////////////////////////////////////////////////////////////////////////
// Aligned types
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(4){
    unsigned char r, g, b, a;
} RGBA8;

typedef unsigned int I32;

typedef struct __align__(8){
    unsigned int l, a;
} LA32;

typedef struct __align__(16){
    unsigned int r, g, b;
} RGB32;

typedef struct __align__(16){
    unsigned int r, g, b, a;
} RGBA32;


////////////////////////////////////////////////////////////////////////////////
// Because G80 class hardware natively supports global memory operations
// only with data elements of 4, 8 and 16 bytes, if structure size
// exceeds 16 bytes, it can't be efficiently read or written,
// since more than one global memory non-coalescable load/store instructions
// will be generated, even if __align__ option is supplied.
// "Structure of arrays" storage strategy offers best performance 
// in general case. See section 5.1.2 of the Programming Guide.
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(16){
    RGBA32 c1, c2;
} RGBA32_2;



////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b){
    return a - a % b;
}



////////////////////////////////////////////////////////////////////////////////
// Simple CUDA kernel.
// Copy is carried out on per-element basis,
// so it's not per-byte in case of padded structures.
////////////////////////////////////////////////////////////////////////////////
template<class TData> __global__ void testKernel(
    TData *d_odata,
    TData *d_idata,
    int numElements
){
    const int        tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;

    for(int pos = tid; pos < numElements; pos += numThreads)
        d_odata[pos] = d_idata[pos];
}



////////////////////////////////////////////////////////////////////////////////
// Validation routine for simple copy kernel.
// We must know "packed" size of TData (number_of_fields * sizeof(simple_type))
// and compare only these "packed" parts of the structure, 
// containig actual user data. The compiler behavior with padding bytes
// is undefined, since padding is merely a placeholder 
// and doesn't contain any user data.
////////////////////////////////////////////////////////////////////////////////
template<class TData> int testCPU(
    TData *h_odata,
    TData *h_idata,
    int numElements,
    int packedElementSize
){
    for(int pos = 0; pos < numElements; pos++){
        TData src = h_idata[pos];
        TData dst = h_odata[pos];
        for(int i = 0; i < packedElementSize; i++)
            if( ((char *)&src)[i] != ((char *)&dst)[i] ) return 0;
    }

    return 1;
}



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//Memory chunk size in bytes. Reused for test
#ifdef __DEVICE_EMULATION__
    const int       MEM_SIZE = 16000000;
    const int NUM_ITERATIONS = 1;
#else
    const int       MEM_SIZE = 50000000;
    const int NUM_ITERATIONS = 32;
#endif

//GPU input and output data
unsigned char *d_idata, *d_odata;
//CPU input data and instance of GPU output data
unsigned char *h_idataCPU, *h_odataGPU;
unsigned int hTimer;



template<class TData> void runTest(int packedElementSize){
    const int totalMemSizeAligned = iAlignDown(MEM_SIZE, sizeof(TData));
    const int         numElements = iDivDown(MEM_SIZE, sizeof(TData));

    //Clean output buffer before current test
    CUDA_SAFE_CALL( cudaMemset(d_odata, 0, MEM_SIZE) );
    //Run test
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
    for(int i = 0; i < NUM_ITERATIONS; i++){
        testKernel<<<64, 256>>>(
            (TData *)d_odata,
            (TData *)d_idata,
            numElements
        );
        CUT_CHECK_ERROR("testKernel() execution failed\n");
    }
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(hTimer) );
    double gpuTime = cutGetTimerValue(hTimer) / NUM_ITERATIONS;
    printf(
        "Avg. time: %f ms / Copy throughput: %f GB/s.\n", gpuTime,
        (double)totalMemSizeAligned / (gpuTime * 0.001 * 1073741824.0)
    );

    //Read back GPU results and run validation
    CUDA_SAFE_CALL( cudaMemcpy(h_odataGPU, d_odata, MEM_SIZE, cudaMemcpyDeviceToHost) );
    int flag = testCPU(
        (TData *)h_odataGPU,
        (TData *)h_idataCPU,
        numElements,
        packedElementSize
    );
    printf(flag ? "TEST PASSED\n" : "TEST FAILED\n" );
}



int main(int argc, char **argv){
    int i;

    CUT_DEVICE_INIT(argc, argv);
    CUT_SAFE_CALL( cutCreateTimer(&hTimer) );

    printf("Allocating memory...\n");
        h_idataCPU = (unsigned char *)malloc(MEM_SIZE);
        h_odataGPU = (unsigned char *)malloc(MEM_SIZE);
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_idata, MEM_SIZE));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_odata, MEM_SIZE));

    printf("Generating host input data array...\n");
        for(i = 0; i < MEM_SIZE; i++)
            h_idataCPU[i] = (i & 0xFF) + 1;
    printf("Uploading input data to GPU memory...\n");
        CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idataCPU, MEM_SIZE, cudaMemcpyHostToDevice) );


    printf("Testing misaligned types...\n");
        printf("uint8...\n");
        runTest<uint8>(1);

        printf("uint16...\n");
        runTest<uint16>(2);

        printf("RGBA8_misaligned...\n");
        runTest<RGBA8_misaligned>(4);

        printf("LA32_misaligned...\n");
        runTest<LA32_misaligned>(8);

        printf("RGB32_misaligned...\n");
        runTest<RGB32_misaligned>(12);

        printf("RGBA32_misaligned...\n");
        runTest<RGBA32_misaligned>(16);

    printf("Testing aligned types...\n");
        printf("RGBA8...\n");
        runTest<RGBA8>(4);

        printf("I32...\n");
        runTest<I32>(4);

        printf("LA32...\n");
        runTest<LA32>(8);

        printf("RGB32...\n");
        runTest<RGB32>(12);

        printf("RGBA32...\n");
        runTest<RGBA32>(16);

        printf("RGBA32_2...\n");
        runTest<RGBA32_2>(32);


    printf("Shutting down...\n");
        CUDA_SAFE_CALL(cudaFree(d_idata));
        CUDA_SAFE_CALL(cudaFree(d_odata));
        free(h_odataGPU);
        free(h_idataCPU);

    CUT_SAFE_CALL( cutDeleteTimer(hTimer) );
    CUT_EXIT(argc, argv);
}
