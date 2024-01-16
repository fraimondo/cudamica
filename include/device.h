/*
W * device.h
 *
 *  Created on: Jul 11, 2012
 *      Author: fraimondo
 */

#ifndef DEVICE_H_
#define DEVICE_H_

#include <tipos.h>
#include <config.h>
#include <curand.h>
#include <cublas.h>
#include <amica.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct launchconfig {
	dim3		nblocks;
	dim3		nthreads;
	natural		ndata;
} launchconfig_t;

typedef struct devicetune {
	 /* ====================
	  * Channels by Samples
	  * ====================
	  */

	/*
	 * Each thread process 1 sample
	 * Each block process N samples
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension
	 */
	struct launchconfig c1c1s_12dim;

	/*
	 * Each thread process 1 sample
	 * Each block process N samples using n threads
	 * blockDim has 1 dimension
	 * gridDim has 1 dimension
	 */
	struct launchconfig c1cNs_11dim;

	/*======================
	 * Channels by blocksize
	 * ======================
	 */
	/*
	 * Each thread process 1 sample
	 * Each block process N samples
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension
	 */
	struct launchconfig c1c1bs_12dim;

	/*======================
	 * Channels by blocksize
	 * ======================
	 */
	/*
	 * Each thread process M samples
	 * Each block process N channels
	 * blockDim has 1 dimension
	 * gridDim has 2 dimension
	 * nsdm = blockIdx.y
	 */
	struct launchconfig c1cNbs_21dim;

	/*======================
	 * Channels by blocksize
	 * ======================
	 */
	/*
	 * Each thread process M samples
	 * Each block process N channels
	 * blockDim has 1 dimension
	 * gridDim has 1 dimension
	 */
	struct launchconfig c1cNbs_11dim;



	/*==============================
	 * Channels by blocksize by nsdm
	 * =============================
	 */
	/*
	 * Each thread process 1 sample
	 * Each block process N samples
	 * blockDim has 2 dimensions
	 * gridDim has 2 dimension.
	 * nsdm = blockIdx.y
	 */
	struct launchconfig c1c1snsdm_22dim;

	/*
	 * Each thread process 1 sample
	 * Each block process N samples
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * nsdm = threadIdx.y
	 */
	struct launchconfig c1c1snsdm_12dim;


	/* ====================
	 * Channels by Channels
	 * ====================
	 */

	/* Each thread process 1 element
	 * Each block process 1 row
  	 * blockDim has 1 dimension
	 * gridDim has 1 dimension
	 */
	struct launchconfig c1c1c_11dim;

	/* ====================
	 * Blocksize by nmmodels
	 * ====================
	 */

	/*
	 * Each thread process 1 element
	 * Each block process nthreads elements
	 * blockDim has 2 dimension by nmmodels
	 * gridDim has 1 dimension
	 */
	struct launchconfig blocksize_12dim;

	/* ====================
	 * Vectors
	 * ====================
	 */

	/*
	 * Each thread process 1 element
	 * Each block process nthreads elements
	 * blockDim has 1 dimension
	 * gridDim has 1 dimension
	 */
	struct launchconfig vblocksize_11dim;

	/*
	 * Each thread process 1 element
	 * Each block process channels elements
	 * blockDim has 1 dimension
	 * gridDim has 1 dimension
	 */
	struct launchconfig vchannels_11dim;



	/* ====================
	 * Channels by nsdm by nmmodels
	 * ====================
	 */
	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.t = nsdm * nmmodels
	 */
	struct launchconfig c1c1snsdmnmmodels_12dim;


	/* ====================
	 * Channels by nmmodels
	 * ====================
	 */
	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.y = nmmodels
	 */
	struct launchconfig c1c1snmmodels_12dim;

	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 1 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.y = nmmodels
	 */
	struct launchconfig c1c1snmmodels_11dim;

	/* ====================
	 * Channels by nsdm
	 * ====================
	 */
	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 1 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.y = nsdm
	 */
	struct launchconfig c1c1snsdm_11dim;


	/* ====================
	 * Channels by channels by nmmodels
	 * ====================
	 */
	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.t = nmmodels
	 */
	struct launchconfig c1c1snchansnmmodels_12dim;

} devicetune;

typedef struct {
	int 					realdevice;
	struct 					cudaDeviceProp deviceProp;
	int						nthreads;
} cudaDevices_t;

typedef struct {
	int 					realdevice;
	struct 					cudaDeviceProp * deviceProp;
	int						nthreads;
	curandGenerator_t 		curandHandle;
	cublasHandle_t			cublasHandle;
	cudaStream_t 			stream;

	/*
	 *
	 */
	struct devicetune 		dimensions;

	/* Device dependant variables */
	cudaPitchedPtr			data;		//real: nchannels by nsamples
	cudaPitchedPtr			sphere;		//real: nchannels by nchannels
	cudaPitchedPtr			eigd;		//TODO: find where I use this!
	cudaPitchedPtr			eigv;		//real: nchannels by nchannels
	cudaPitchedPtr			means;		//real: nchannels
	real					ldetS;


	/* Device working variables (initialized in device.cu) */
	cudaPitchedPtr			rnxnwork;	/* real: nchannels by nchannels by nmmodels */
	//TODO: Check that nchannels is greater than nsdm
	//TODO: Check that nmmodels is less than blocksize
	//TODO: Check that nchannels is less than blocksize

	cudaPitchedPtr			rbswork;	/* real: blocksize */
	cudaPitchedPtr			rbsxMwork;	/* real: blocksize by nmmodels */
	cudaPitchedPtr			rnwork;		/* real: nchannels */
	cudaPitchedPtr			rnxbswork;	/* real: nchannels by blocksize */
	cudaPitchedPtr			rnxbswork2;	/* real: nchannels by blocksize */
	cudaPitchedPtr			rnxbswork3;	/* real: nchannels by blocksize */

	cudaPitchedPtr			b;			/* real: nchannels by blocksize by nmmodels */
	cudaPitchedPtr			y;			/* real: nchannels by blocksize by nsdm by nmmodels */
	cudaPitchedPtr			Q;			/* real: nchannels by blocksize by nsdm by nmmodels */
	cudaPitchedPtr			z;			/* real: nchannels by blocksize by nsdm by nmmodels */

	cudaPitchedPtr			v;			/* real: blocksize by nmmodels */

	cudaPitchedPtr			g;			/* real: nchannels by blocksize */
	cudaPitchedPtr			u;			/* real: nchannels by blocksize by nsdm */

	cudaPitchedPtr			fp;			/* real: nchannels by blocksize by nsdm */
	cudaPitchedPtr			ufp;		/* real: nchannels by blocksize by nsdm */

	/* Copies of model variables */
	cudaPitchedPtr			c;			/* real: nchannels by nmmodels*/
	cudaPitchedPtr			beta;		/* real: nchannels by nsdm by nmmodels */
	cudaPitchedPtr			mu;			/* real: nchannels by nsdm by nmmodels */
	cudaPitchedPtr			alpha;		/* real: nchannels by nsdm by nmmodels */
	cudaPitchedPtr			rho;		/* real: nchannels by nsdm by nmmodels */

	cudaPitchedPtr			Lt;			/* real: blocksize by nmmodels */


	/* redefinitions */
#define		W 	rnxnwork
#define 	usumwork fp
#define		B rnxnwork
#define 	noms rnxbswork3
#define 	denoms fp
#define		atmp g
#define		acopy b
#define 	norms rnxbswork


} device_t;


error		getDevices(void);
error 		selectDevice(natural * deviceNums, natural count);
void 		printCapabilities(cudaDeviceProp* properties);
error		initializeDevs(void);
error		finalizeDevs(void);

extern 		natural		gpuCount;
extern		device_t 	gpus[MAX_DEVS];

#ifdef __cplusplus
}
#endif



#endif /* DEVICE_H_ */
