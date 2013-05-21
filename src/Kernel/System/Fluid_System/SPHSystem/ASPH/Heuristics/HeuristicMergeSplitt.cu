#include <HeuristicMergeSplitt.cuh>

extern "C"
{
/****************************************************************************************************/
/*********************** DEPTH HEURISTIC ************************************************************/
/****************************************************************************************************/
	void evaluateHeuristic_Depth_Kernel(float3 *pos, float depthMax, bool* res)
	{
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		float3 position = pos[index];
		if(position.z<=depthMax)
			res[index] = true;
		else
			res[index] = false;
	}
/****************************************************************************************************/
	void evaluateHeuristic_Depth(float *pos, float depthMax, uint nbBodies, bool* res)
	{

		int numThreadsX, numBlocksX;
    		computeGridSize(numBodies,256, numBlocksX, numThreadsX);
    		dim3 dimBlock(numBlocksX,1);
    		dim3 dimGrid(numThreadsX,1);
		evaluateHeuristic_Depth_Kernel<<< dimGrid, dimBlock>>>((float3*) pos, depthMax, res);
		CUT_CHECK_ERROR("evaluate heuristic depth kernel execution failed");
	}
/****************************************************************************************************/
/************************    VIEWER DISTANCE KERNEL *************************************************/
/****************************************************************************************************/
	void evaluateHeuristic_ViewerDistance_Kernel(float3 *pos, float3 viewer, float distanceMax, bool* res)
	{
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		float3 pos2Viewer = pos[index] - viewer;
		if(length(pos2Viewer)>=distanceMax)
			res[index] = true;
		else
			res[index] = false;
	}
/****************************************************************************************************/
	void evaluateHeuristic_ViewerDistance(float *pos, Vector3 viewer, float distanceMax, uint nbBodies, bool* res)
	{

		int numThreadsX, numBlocksX;
    		computeGridSize(numBodies,256, numBlocksX, numThreadsX);
    		dim3 dimBlock(numBlocksX,1);
    		dim3 dimGrid(numThreadsX,1);
		float3 O = make_float3(viewer.x(),viewer.y(),viewer.z());
		evaluateHeuristic_ViewerDistance_Kernel<<< dimGrid, dimBlock>>>((float3*) pos, O, distanceMax, res);
		CUT_CHECK_ERROR("evaluate heuristic viewer distance kernel execution failed");
	}
/****************************************************************************************************/
/*********************** MAGNITUDE FORCES HEURISTIC ************************************************************/
/****************************************************************************************************/
	void evaluateHeuristic_ForceMagnitude_Kernel(float3 *forces, float magnitudeMax, bool* res)
	{
		int index = blockIdx.x*blockDim.x + threadIdx.x;
		float3 force = forces[index];
		if(length(force)>=magnitudeMax)
			res[index] = true;
		else
			res[index] = false;
	}
/****************************************************************************************************/
	void evaluateHeuristic_ForceMagnitude(float *forces, float magnitudeMax, uint nbBodies, bool* res)
	{

		int numThreadsX, numBlocksX;
    		computeGridSize(numBodies,256, numBlocksX, numThreadsX);
    		dim3 dimBlock(numBlocksX,1);
    		dim3 dimGrid(numThreadsX,1);
		evaluateHeuristic_ForceMagnitude_Kernel<<< dimGrid, dimBlock>>>((float3*) forces, magnitudeMax, res);
		CUT_CHECK_ERROR("evaluate heuristic magnitude force kernel execution failed");
	}
}
