/*
 * error.h
 *
 *  Created on: Jul 11, 2012
 *      Author: fraimondo
 */

#ifndef ERROR_H_
#define ERROR_H_

#include <tipos.h>
#include <curand.h>
#include <cublas.h>
#include <magma.h>
#include <tools.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "ERROR::%s at line %d in file %s\n",				\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDA_CHECK_LAST() {													\
	cudaError_t _m_cudaStat = cudaGetLastError();							\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "ERROR::%s at line %d in file %s\n",				\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }



#define CUDA_KERNEL_CALL(function, blocks, threads, mem, stream, ...) { 	\
	PRINT_KERNEL_CALL(function, blocks, threads, mem, stream, __VA_ARGS__); \
	function<<<blocks, threads, mem, stream>>>(__VA_ARGS__);				\
	cudaError_t _m_cudaStat = cudaGetLastError();							\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "ERROR::%s at line %d in file %s\n",				\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


/**
 * This macro checks return value of the C call and exits
 * the application if the call failed.
 */
#define C_CHECK_RETURN(value) {												\
	natural stat = value;													\
	if (stat != SUCCESS) {													\
		fprintf(stderr, "ERROR::%s at line %d in file %s\n",				\
				cGetErrorString(stat), __LINE__, __FILE__);					\
		exit(1);															\
	} }

/**
 * This macro checks return value of the MKL call and exits
 * the application if the call failed.
 */
#define MKL_CHECK_RETURN(value) {												\
	int stat = value;													\
	if (stat != SUCCESS) {													\
		fprintf(stderr, "ERROR::%s at line %d in file %s\n",				\
				mklGetErrorString(stat), __LINE__, __FILE__);					\
		exit(1);															\
	} }

/**
 * This macro checks return value of the Magma call and exits
 * the application if the call failed.
 */
#define MAGMA_CHECK_RETURN(value) {												\
	magma_int_t stat = value;													\
	if (stat != SUCCESS) {													\
		fprintf(stderr, "ERROR::%s at line %d in file %s\n",				\
				magmaGetErrorString(stat), __LINE__, __FILE__);					\
		exit(1);															\
	} }

/**
 * This macro checks return value of the CURAND runtime call and exits
 * the application if the call failed.
 */
#define CURAND_CHECK_RETURN(value) {											\
	curandStatus _m_curandStat = value;										\
	if (_m_curandStat != CURAND_STATUS_SUCCESS) {										\
		fprintf(stderr, "ERROR::%s at line %d in file %s\n",				\
				curandGetErrorString(_m_curandStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/**
 * This macro checks return value of the CUBLAS runtime call and exits
 * the application if the call failed.
 */
#define CUBLAS_CHECK_RETURN(value) {											\
	cublasStatus_t _m_cublasStat = value;										\
	if (_m_cublasStat != CUBLAS_STATUS_SUCCESS) {										\
		fprintf(stderr, "ERROR::%s at line %d in file %s\n",				\
				cublasv2GetErrorString(_m_cublasStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }



#define SUCCESS 0
#define ERRORNODEVICEMEM 	-1				//No more memory on device
#define ERRORNOPARAM 		-2				//Parameter error
#define ERRORINVALIDPARAM	-3				//Parameter is invalid
#define ERRORINVALIDCONFIG	-4				//Config file is invalid
#define ERRORNOFILE			-5				//Error opening file
#define ERRORIO				-6				//Input/Output error
#define ERRORNOMEM			-7				//No more memory on host

#ifdef __cplusplus
extern "C" {
#endif
	const char * cGetErrorString(natural value);
	const char * mklGetErrorString(int value);
	const char * curandGetErrorString(curandStatus_t value);
	const char * cublasv2GetErrorString(cublasStatus_t value);
	const char * magmaGetErrorString(magma_err_t value);
#ifdef __cplusplus
}
#endif


#endif /* ERROR_H_ */
