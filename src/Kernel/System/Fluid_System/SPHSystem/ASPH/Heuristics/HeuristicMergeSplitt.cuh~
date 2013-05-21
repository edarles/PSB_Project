#include <common.cuh>
#include <Vector3.h>
#include <cutil_math.cuh>

extern "C"
{
/****************************************************************************************************/
	void evaluateHeuristic_Depth_Kernel(float3 *pos, float depthMax, bool* res);
	void evaluateHeuristic_Depth(float* pos, float depthMax, uint nbBodies, bool* res);
/****************************************************************************************************/
/****************************************************************************************************/
	void evaluateHeuristic_ViewerDistance_Kernel(float3 *pos, float3 viewer, float distanceMax, bool* res);
	void evaluateHeuristic_ViewerDistance(float* pos, Vector3 viewer, float distanceMax, uint nbBodies, bool* res);
/****************************************************************************************************/
/****************************************************************************************************/
	void evaluateHeuristic_ForceMagnitude_Kernel(float3 *forces, float magnitudeMax, bool* res);
	void evaluateHeuristic_ForceMagnitude(float* forces, float magnitudeMax, uint nbBodies, bool* res);
/****************************************************************************************************/
/****************************************************************************************************/

}
