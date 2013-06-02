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

/* Computation of eigenvalues of a small symmetric, tridiagonal matrix */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
#include "cutil.h"
#include "config.h"
#include "structs.h"
#include "matlab.h"

// includes, kernels
#include "bisect_kernel_small.cu"

// includes, file
#include "bisect_small.cuh"

////////////////////////////////////////////////////////////////////////////////
//! Determine eigenvalues for matrices smaller than MAX_SMALL_MATRIX
//! @param TimingIterations  number of iterations for timing
//! @param  input  handles to input data of kernel
//! @param  result handles to result of kernel
//! @param  mat_size  matrix size
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
//! @param  precision  desired precision of eigenvalues
//! @param  iterations  number of iterations for timing 
////////////////////////////////////////////////////////////////////////////////
void
computeEigenvaluesSmallMatrix( const InputData& input, ResultDataSmall& result, 
                               const unsigned int mat_size,
                               const float lg, const float ug,
                               const float precision,
                               const unsigned int iterations ) 
{
  unsigned int timer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timer));

  CUT_SAFE_CALL( cutStartTimer( timer));
  for( unsigned int i = 0; i < iterations; ++i) {
     
    dim3  blocks( 1, 1, 1);
    dim3  threads( MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

    bisectKernel<<< blocks, threads >>>( input.g_a, input.g_b, mat_size,
                                         result.g_left, result.g_right, 
                                         result.g_left_count, 
                                         result.g_right_count,
                                         lg, ug, 0, mat_size, 
                                         precision 
                                       );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize());
  CUT_SAFE_CALL( cutStopTimer( timer));
  CUT_CHECK_ERROR( "Kernel launch failed");
  printf( "Average time: %f ms (%i iterations)\n", 
          cutGetTimerValue( timer) / (float) iterations, iterations );

  CUT_SAFE_CALL( cutDeleteTimer( timer));
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for the result for small matrices
//! @param result  handles to the necessary memory
//! @param  mat_size  matrix_size
////////////////////////////////////////////////////////////////////////////////
void 
initResultSmallMatrix( ResultDataSmall& result, const unsigned int mat_size) {

  result.mat_size_f = sizeof(float) * mat_size;
  result.mat_size_ui = sizeof(unsigned int) * mat_size;

  result.eigenvalues = (float*) malloc( result.mat_size_f);

  // helper variables
  result.zero_f = (float*) malloc( result.mat_size_f);
  result.zero_ui = (unsigned int*) malloc( result.mat_size_ui);
  for( unsigned int i = 0; i < mat_size; ++i) {

    result.zero_f[i] = 0.0f;
    result.zero_ui[i] = 0;
    
    result.eigenvalues[i] = 0.0f;
  }

  CUDA_SAFE_CALL( cudaMalloc( (void**) &result.g_left, result.mat_size_f));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &result.g_right, result.mat_size_f));

  CUDA_SAFE_CALL( cudaMalloc( (void**) &result.g_left_count, 
                              result.mat_size_ui));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &result.g_right_count, 
                              result.mat_size_ui));

  // initialize result memory
  CUDA_SAFE_CALL( cudaMemcpy( result.g_left, result.zero_f, result.mat_size_f,
                              cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL( cudaMemcpy( result.g_right, result.zero_f, result.mat_size_f,
                              cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL( cudaMemcpy( result.g_right_count, result.zero_ui, 
                              result.mat_size_ui,
                              cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL( cudaMemcpy( result.g_left_count, result.zero_ui, 
                              result.mat_size_ui,
                              cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup memory and variables for result for small matrices
//! @param  result  handle to variables
////////////////////////////////////////////////////////////////////////////////
void 
cleanupResultSmallMatrix( ResultDataSmall& result) {

  freePtr( result.eigenvalues);
  freePtr( result.zero_f);
  freePtr( result.zero_ui);

  CUDA_SAFE_CALL( cudaFree( result.g_left));
  CUDA_SAFE_CALL( cudaFree( result.g_right));
  CUDA_SAFE_CALL( cudaFree( result.g_left_count));
  CUDA_SAFE_CALL( cudaFree( result.g_right_count));
}

////////////////////////////////////////////////////////////////////////////////
//! Process the result obtained on the device, that is transfer to host and
//! perform basic sanity checking
//! @param  input  handles to input data
//! @param  result  handles to result data
//! @param  mat_size   matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
void
processResultSmallMatrix( const InputData& input, const ResultDataSmall& result,
                          const unsigned int mat_size,
                          const char* filename ) {

  const unsigned int mat_size_f = sizeof(float) * mat_size;
  const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

  // copy data back to host
  float* left = (float*) malloc( mat_size_f);
  unsigned int* left_count = (unsigned int*) malloc( mat_size_ui);

  CUDA_SAFE_CALL( cudaMemcpy( left, result.g_left, mat_size_f, 
                              cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL( cudaMemcpy( left_count, result.g_left_count, mat_size_ui, 
                              cudaMemcpyDeviceToHost));

  float* eigenvalues = (float*) malloc( mat_size_f);

  for( unsigned int i = 0; i < mat_size; ++i) {
      eigenvalues[left_count[i]] = left[i];
  }

  // save result in matlab format
  writeTridiagSymMatlab( filename, input.a, input.b+1, eigenvalues, mat_size);

  freePtr( left);
  freePtr( left_count);
  freePtr( eigenvalues);
}

