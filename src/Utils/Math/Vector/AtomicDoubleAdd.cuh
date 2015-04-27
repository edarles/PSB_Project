extern "C"
{
__device__ inline int floatToOrderedInt(float floatVal)
{
	int intVal = __float_as_int( floatVal );
	return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ inline float orderedIntToFloat(int intVal)
{
	return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );
} 

__device__ inline float atomicFloatMin(float *addr, float value)
{
	float old = *addr, assumed;
	if(old <= value) return old;
	do {
		assumed = old;
		old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

	}
	while(old!=assumed);
	return old;
}

__device__ inline void atomicFloatMax(float *address, float value)
{
	int oldval, newval, readback;
    	oldval = __float_as_int(*address);
    	newval = __float_as_int(value);
	if(oldval<newval){
    		while ((readback=atomicCAS((int*)address, oldval, newval)) != oldval)
    		{
        		oldval = readback;
        		newval = __float_as_int(value);
    		}
	}
}

__device__ inline void atomicDoubleMin (double *address, double value)
{
   unsigned long long oldval, newval, readback; 
 
   oldval = __double_as_longlong(*address);
   newval = __double_as_longlong(__longlong_as_double(oldval) + value);
   while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
     {
      oldval = readback;
      if(__longlong_as_double(oldval)>value)
	newval = __double_as_longlong(value);
     }
}

__device__ inline void atomicDoubleMax (double *address, double value)
{
   unsigned long long oldval, newval, readback; 
 
   oldval = __double_as_longlong(*address);
   newval = __double_as_longlong(__longlong_as_double(oldval) + value);
   while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
     {
      oldval = readback;
      if(__longlong_as_double(oldval)<value)
	newval = __double_as_longlong(value);
     }
}

__device__ inline void atomicDoubleAdd(double *address, double value)  //See CUDA official forum
 {
    unsigned long long oldval, newval, readback;
    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
        oldval = readback;
        newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
}

__device__ inline void atomicDoubleExch(double *address, double value)  //See CUDA official forum
 {
    unsigned long long oldval, newval, readback;
 
    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(value);
    while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
        oldval = readback;
        newval = __double_as_longlong(value);
    }
}

__device__ inline void atomicDouble3Add(double3 *address, double3 value)  //See CUDA official forum
 {
    atomicDoubleAdd(&address->x,value.x);
    atomicDoubleAdd(&address->y,value.y);
    atomicDoubleAdd(&address->z,value.z);
 }

__device__ inline void atomicDouble3Exch(double3 *address, double3 value)  //See CUDA official forum
 {
    atomicDoubleExch(&address->x,value.x);
    atomicDoubleExch(&address->y,value.y);
    atomicDoubleExch(&address->z,value.z);
 }
 
inline void SerialAtomicAdd(double *address, double value)
 {
  *address = *address + value;
  return;
 }
 
inline void SerialAtomicMin(int *address, int value)
 {
   if (value < *address)
     *address = value;
   return;
 }
 
  inline void SerialAtomicMax(int *address, int value)
 {
   if (value > *address)
     *address = value;
   return;
 }
 
}
