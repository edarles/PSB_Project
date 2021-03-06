#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
//#include </usr/local/NVIDIA_GPU_Computing_SDK/C/src/simplePrintf/cuPrintf.cuh>

#include <ForceExt_Trochoide.cuh>
#include <ForceExt_Trochoide.h>
#include <ForceExt_Constante.cuh>
#include <ForceExt_Constante.h>

#include <common.cuh>
#include <Vector3.h>

/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
extern "C"
{

/**************************************************************************************************************************************/
void evaluate_ForceExt_Constante(uint nbBodies, double* accumForceBuffer, double* mass,
			         ForceExt_Constante *FC)
{
    int numBlocksX, numThreadsX;
    computeGridSize(nbBodies, numBlocksX, numThreadsX);
    float3 direction = make_float3(FC->getDirection().x(),FC->getDirection().y(),FC->getDirection().z());
    float  amplitude = FC->getAmplitude();

    evaluate_force_constante<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*)accumForceBuffer,mass,direction,amplitude);
    cudaDeviceSynchronize();
}

/**************************************************************************************************************************************/
void evaluate_ForceExt_Trochoide(uint nbBodies, double* oldPos, double* accumForceBuffer, double* mass,
				 ForceExt_Trochoide *T)
{
    float A = T->getAmplitude();
    float k = T->getNbreOnde();
    float theta = T->getDeviation();
    float w = T->getPulsation();
    float phi = T->getDephasage();
    float time = T->getTime();

    int numBlocksX, numThreadsX;
    computeGridSize(nbBodies, numBlocksX, numThreadsX);
  
    evaluate_force_trochoide<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*)oldPos, (double3*)accumForceBuffer, mass, A, k, theta, w, phi, time);
    cudaDeviceSynchronize();
}

}
