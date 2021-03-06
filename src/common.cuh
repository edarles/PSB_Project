extern "C" {

int getcudaEnabledDeviceCount();

bool cudaInit(int argc, char **argv);

void cudaGLInit(int argc, char **argv);

void allocateArray(void **devPtr, size_t size);

void freeArray(void *devPtr);

void threadSync();

void copyArrayToDevice(void *device, const void *host, int offset, int size);

void copyArrayDeviceToDevice(void *device, const void *host, int offset, int size);

void copyArrayFromDevice(void *host, const void *device,struct cudaGraphicsResource **cuda_vbo_resource, int size);

int iDivUp(int a, int b);

void computeGridSize(int blockSize, int &numBlocks, int &numThreads);

}
