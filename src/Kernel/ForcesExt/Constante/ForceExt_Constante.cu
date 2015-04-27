#include <ForceExt_Constante.cuh>
#include <ForceExt_Constante_Kernel.cuh>

extern "C"
{
/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
void evaluate_force_constante_CUDA(double* accumForceBuff, double* mass, Vector3 direction, float amplitude, uint nbBodies)
{
    int numBlocksX, numThreadsX;
    computeGridSize(nbBodies, numBlocksX, numThreadsX);
    evaluate_force_constante_Kernel<<<numBlocksX,numThreadsX>>>((double3*)accumForceBuff,mass,
							        make_float3(direction.x(),direction.y(),direction.z()),amplitude,nbBodies);
}
/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
}
