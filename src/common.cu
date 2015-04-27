
#define CUDA

#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <common.cuh>

#if defined (CUDA)
	#include <cuda_runtime_api.h>
	#include <cuda.h>
	#include <device_launch_parameters.h>
	#include <cuda_gl_interop.h>
	#include <cuda_runtime_api.h>
	#include <helper_functions.h>
	#include <helper_cuda.h>
	#include <helper_cuda_gl.h>
	#include <cuda_runtime.h>
#endif

extern "C"
{

  #if defined (CUDA)
  int getCudaEnabledDeviceCount()
  {
    int count;
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetDeviceCount( &count );
    if (error == cudaErrorNoDevice)
    {
	printf("NO DEVICE\n");
        count = 0;
    }
    else if (error == cudaErrorInsufficientDriver)
    {
        count = -1;
    }
   // else //should never happen
    {
        checkCudaErrors(error);
    }
    return count;
  }

  bool cudaInit(int argc, char **argv)
  {
       int count = getCudaEnabledDeviceCount();

       int devID = findCudaGLDevice(argc, (const char **)argv);
       
       if (devID < 0)
       {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
	    return false;
        }
	else {
		 cudaDeviceProp deviceProp;
		 checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
       		 printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
		 return true;
	}
    }

    void cudaGLInit(int argc, char **argv)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        findCudaGLDevice(argc, (const char **)argv);
    }
 
    void allocateArray(void **devPtr, size_t size)
    {
        cudaMalloc(devPtr, size);
    }

    void freeArray(void *devPtr)
    {
        cudaFree(devPtr);
    }

    void threadSync()
    {
        cudaDeviceSynchronize();
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice);
    }

    void copyArrayDeviceToDevice(void *device, const void *host, int offset, int size)
    {
        cudaMemcpy((char *) device + offset, host, size, cudaMemcpyDeviceToDevice);
    }

    void copyArrayDeviceToDevice2(void *device, const void *host, int offset, int size)
    {
        cudaMemcpy(device, (char *) host + offset, size, cudaMemcpyDeviceToDevice);
    }

    void copyArrayFromDevice(void *device, const void *host, int offset, int size)
    {
        cudaMemcpy((char *) device + offset, host, size, cudaMemcpyDeviceToHost);
    }
    int iDivUp(int a, int b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(int n, int &numBlocks, int &numThreads)
    {
	int blockSize = 512;
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
	//printf("nbBlocks:%d nbThreads:%d\n",numBlocks,numThreads);
    }
    #endif
}
