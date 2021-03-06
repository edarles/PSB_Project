/**
 * @file double3.h
 * 3D vector operations
 */

#ifndef _DOUBLE3_H
#define _DOUBLE3_H

/**
 * calculates the scalar (dot) product of two vectors
 * @param[in] a first vector
 * @param[in] b second vector
 * @return the dot product
 */
inline __device__ __host__ double dot(const double3 & a, const double3 & b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * calculates the vector (cross) product of two vectors
 * @param[in] a first vector
 * @param[in] b second vector
 * @return the dot product
 */
inline __device__ __host__ double3 cross(const double3 & a, const double3 & b) {
  return make_double3(a.y*b.z - a.z*b.y,
                     a.z*b.x - a.x*b.z,
                     a.x*b.y - a.y*b.x);
}

/**
 * calculates the squared length of a vector
 * @param a the vector
 * @return squared length
 */
inline __device__ __host__ double abs2(const double3 & a) {
  return dot(a,a);
}

/**
 * calculates the length of a vector
 * @param a the vector
 * @return length
 */
inline __device__ __host__ double length(const double3 & a) {
  return sqrt(abs2(a));
}

/**
 * calculates the length of a vector
 * @param a the vector
 * @return length
 */
inline __device__ __host__ double3 normalize(const double3 & a) {
  double l = length(a);
  return make_double3(a.x/l,a.y/l,a.z/l);
}


/**
 * operator to subtract two vectors
 * @param[in] a first vector
 * @param[in] b second vector
 * @return a-b
 */
inline __device__ __host__ double3 operator-(const double3 & a, const double3 & b) {
  return make_double3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ __host__ double3 operator-(const float3 & a, const double3 & b) {
  return make_double3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ __host__ double3 operator-(const double3 & a, const float3 & b) {
  return make_double3(a.x-b.x, a.y-b.y, a.z-b.z);
}
/**
 * operator to add two vectors
 * @param[in] a first vector
 * @param[out] b second vector
 * @return a+b
 */
inline __device__ __host__ double3 operator+(const double3 & a, const double3 & b) {
  return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ __host__ double3 operator+(const double3 & a, const float3 & b) {
  return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ __host__ double3 operator+(const float3 & a, const double3 & b) {
  return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}
/**
 * negates a vector
 * @param[in] a the vector
 * @return -a
 */
inline __device__ __host__ double3 operator-(const double3 & a) {
  return make_double3(-a.x,-a.y,-a.z);
}

/**
 * scales a vector (multiplication with a scalar)
 * @param[in] a the vector
 * @param[in] b the scalar
 * @return a*b
 */
inline __device__ __host__ double3 operator*(const double3 & a, double b) {
  return make_double3(a.x*b, a.y*b, a.z*b);
}

/**
 */
inline __device__ __host__ double3 operator+=(const double3 & a, double3 b) {
  return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

/**
 */
inline __device__ __host__ double3 operator+=(const double3 & a, float3 b) {
  return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

/**
 * scales a vector (multiplication with a scalar)
 * @param[in] a the vector
 * @param[in] b the scalar
 * @return a*b
 */
inline __device__ __host__ double3 operator*(double b, const double3 & a) {
  return make_double3(a.x*b, a.y*b, a.z*b);
}

inline __device__ __host__ double3 operator*= (double3 a, const float & b) {
  return make_double3(a.x*b, a.y*b, a.z*b);
}

/**
 * scales a vector (division by a scalar)
 * @param[in] a the vector
 * @param[in] b the scalar
 * @return a/b
 */
inline __device__ __host__ double3 operator/(const double3 & a, double b) {
  b = 1.0f / b;
  return a*b;
}

inline __device__ __host__ double3 operator/=(const double3 & a, double b) {
  b = 1.0f / b;
  return a*b;
}

#endif  /* _DOUBLE3_H */

