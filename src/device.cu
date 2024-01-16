
#include <device.h>
#include <error.h>
#include <config.h>
#include <stdio.h>
#include <tools.h>
#include <helpers.h>


device_t 		gpus[MAX_DEVS];	// In use
natural			gpuCount;

cudaDevices_t 	capables[MAX_DEVS]; // In system
int 			deviceCount;
int				currentDevice;

error getDevices() {
	CUDA_CHECK_RETURN(cudaGetDeviceCount((int*)&deviceCount));
	natural numdev = 0;
	PRINT_LINE();
	fprintf(stdout, "List of cuda devices:\n");
	PRINT_LINE();
	for (numdev = 0; numdev < deviceCount; ++numdev) {
		CUDA_CHECK_RETURN(cudaGetDeviceProperties(&capables[numdev].deviceProp, numdev));
		if (capables[numdev].deviceProp.major == 9999 && capables[numdev].deviceProp.minor == 9999) {
			DPRINTF(1,"Device %lu does not support CUDA\n", numdev);
		} else {
			DPRINTF(1,"Device %lu supports CUDA\n", numdev);
			fprintf(stdout, "Device: %lu\n", numdev);
			printCapabilities(&capables[numdev].deviceProp);
		}
		if (capables[numdev].deviceProp.major == 2) {
			capables[numdev].nthreads = 32;
			} else {
				capables[numdev].nthreads = 8;
		}

	}

	PRINT_LINE();
	PRINT_NEWLINE();
	return SUCCESS;

}

/*
 * Selects the specified cuda device
 *
 * deviceNum: number of the desired device
 */
error selectDevice(natural * deviceNums, natural count) {
	natural deviceNum;
	gpuCount = count;
	for (int i = 0; i < count; i++) {
		deviceNum = deviceNums[i];
		if (deviceNum >= deviceCount) {
			fprintf(stderr, "Selecting invalid device (called getDevices()?)\n");
			return ERRORINVALIDPARAM;
		}
		fprintf(stdout, "\n\nSelecting device %lu", deviceNum);
		CUDA_CHECK_RETURN(cudaSetDevice(deviceNum));
		CUDA_CHECK_RETURN(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
		gpus[i].realdevice = deviceNum;
		gpus[i].deviceProp = &capables[deviceNum].deviceProp;
		gpus[i].nthreads = capables[deviceNum].nthreads;

#ifdef USEP2P
		int can;

		for (int j = 0; j < count; j++) {
			if (i != j) {
				DPRINTF(1, "Testing peer access for device %d\n", deviceNums[j]);
				CUDA_CHECK_RETURN(cudaDeviceCanAccessPeer(&can, deviceNum, deviceNums[j]));
				fprintf(stdout, "Device %d can access %d? %s\n", deviceNum, deviceNums[j], can == 1 ? "Yes" : "No");
				DPRINTF(1, "Enabling peer acces for device %d\n", deviceNums[j]);
				CUDA_CHECK_RETURN(cudaDeviceEnablePeerAccess(deviceNums[j], 0));
			}
		}
#endif
		currentDevice = deviceNum;
		fprintf(stdout, " Success!\n");
#ifndef USEP2P
		DPRINTF(1, "Not using P2P\n")
#endif

	}
	return SUCCESS;
}


/*
 * Prints the device capabilities
 */
void printCapabilities(cudaDeviceProp* properties) {
	fprintf(stdout, "CUDA Device capabilities:\n");
	fprintf(stdout, "	Name: %s\n", properties->name);
	fprintf(stdout, "	Global Mem: %lu\n", properties->totalGlobalMem);
	fprintf(stdout, "	Mem: %lu\n", properties->totalGlobalMem);
	fprintf(stdout, "	Shared Mem per Block: %lu\n", properties->sharedMemPerBlock);
	fprintf(stdout, "	Regs per Block: %d\n", properties->regsPerBlock);
	fprintf(stdout, "	Warp size: %d\n", properties->warpSize);
	fprintf(stdout, "	Mem pitch: %lu\n", properties->memPitch);
	fprintf(stdout, "	Max Threads per Block: %d\n", properties->maxThreadsPerBlock);
	fprintf(stdout, "	Max Threads per Multiprocessor: %d\n", properties->maxThreadsPerMultiProcessor);
	fprintf(stdout, "	Max Threads Dim: %d x %d x %d\n",
		properties->maxThreadsDim[0],
		properties->maxThreadsDim[1],
		properties->maxThreadsDim[2]);
	fprintf(stdout, "	Max Grid Size: %d x %d x %d\n",
		properties->maxGridSize[0],
		properties->maxGridSize[1],
		properties->maxGridSize[2]);
	fprintf(stdout, "	Max Surface 1D: %d\n", properties->maxSurface1D);
	fprintf(stdout, "	Max Surface 1D Layered: %d x %d\n",
		properties->maxSurface1DLayered[0],
		properties->maxSurface1DLayered[1]);
	fprintf(stdout, "	Max Surface 2D: %d x %d\n",
		properties->maxSurface2D[0],
		properties->maxSurface2D[1]);
	fprintf(stdout, "	Max Surface 2D Layered: %d x %d x %d\n",
		properties->maxSurface2DLayered[0],
		properties->maxSurface2DLayered[1],
		properties->maxSurface2DLayered[2]);
	fprintf(stdout, "	Max Surface 3D: %d x %d x %d\n",
		properties->maxSurface3D[0],
		properties->maxSurface3D[1],
		properties->maxSurface3D[2]);
	fprintf(stdout, "	Max Surface Cubemap: %d\n", properties->maxSurfaceCubemap);
	fprintf(stdout, "	Max Surface Cubemap Layered: %d x %d\n",
		properties->maxSurfaceCubemapLayered[0],
		properties->maxSurfaceCubemapLayered[1]);
	fprintf(stdout, "	Max Texture 1D: %d\n", properties->maxTexture1D);
	fprintf(stdout, "	Max Texture 1D Layered: %d x %d\n",
		properties->maxTexture1DLayered[0],
		properties->maxTexture1DLayered[1]);
	fprintf(stdout, "	Max Texture 1D Linear: %d\n", properties->maxTexture1DLinear);
	fprintf(stdout, "	Max Texture 1D Mipmap: %d\n", properties->maxTexture1DMipmap);
	fprintf(stdout, "	Max Texture 2D: %d x %d\n",
		properties->maxTexture2D[0],
		properties->maxTexture2D[1]);
	fprintf(stdout, "	Max Texture 2D Gather: %d x %d\n",
		properties->maxTexture2DGather[0],
		properties->maxTexture2DGather[1]);
	fprintf(stdout, "	Max texture 2D Layered: %d x %d x %d\n",
		properties->maxTexture2DLayered[0],
		properties->maxTexture2DLayered[1],
		properties->maxTexture2DLayered[2]);
	fprintf(stdout, "	Max texture 2D Linear: %d x %d x %d\n",
		properties->maxTexture2DLinear [0],
		properties->maxTexture2DLinear [1],
		properties->maxTexture2DLinear [2]);
	fprintf(stdout, "	Max Texture 2D Mipmap: %d x %d\n",
		properties->maxTexture2DMipmap[0],
		properties->maxTexture2DMipmap[1]);
	fprintf(stdout, "	Max texture 3D: %d x %d x %d\n",
		properties->maxTexture3D [0],
		properties->maxTexture3D [1],
		properties->maxTexture3D [2]);
	fprintf(stdout, "	Max Texture Cubemap: %d\n", properties->maxTextureCubemap);
	fprintf(stdout, "	Max Texture Cubemap Layered: %d x %d\n",
		properties->maxTextureCubemapLayered [0],
		properties->maxTextureCubemapLayered [1]);

	fprintf(stdout, "	Total Const Mem: %lu\n", properties->totalConstMem);
	fprintf(stdout, "	Major: %d\n", properties->major);
	fprintf(stdout, "	Minor: %d\n", properties->minor);
	fprintf(stdout, "	Memory Bus Width: %d\n", properties->memoryBusWidth);
	fprintf(stdout, "	Memory Clock Rate: %d\n", properties->memoryClockRate);
	fprintf(stdout, "	Clock Rate: %d\n", properties->clockRate);
	fprintf(stdout, "	Texture Alignment: %lu\n", properties->textureAlignment);
	fprintf(stdout, "	Device Overlap: %d\n", properties->deviceOverlap);
	fprintf(stdout, "	Multiprocessor Count: %d\n", properties->multiProcessorCount);
	fprintf(stdout, "	Kernel Exec Timeout Enabled: %d\n", properties->kernelExecTimeoutEnabled);
	fprintf(stdout, "	Integrated: %d\n", properties->integrated);
	fprintf(stdout, "	Can Map host mem: %d\n", properties->canMapHostMemory);
	fprintf(stdout, "	Compute mode: %d\n", properties->computeMode);
	fprintf(stdout, "	Concurrent kernels: %d\n", properties->concurrentKernels);
	fprintf(stdout, "	ECC Enabled: %d\n", properties->ECCEnabled);
	fprintf(stdout, "	PCI Bus ID: %d\n", properties->pciBusID);
	fprintf(stdout, "	PCI Device ID: %d\n", properties->pciDeviceID);
	fprintf(stdout, "	TCC Driver: %d\n", properties->tccDriver);
	fprintf(stdout, "	Async Engine Count: %d\n", properties->asyncEngineCount);
	fprintf(stdout, "	L2 Cache Size: %d\n", properties->l2CacheSize);
	fprintf(stdout, "	Surface Alignment: %lu\n", properties->surfaceAlignment);
	fprintf(stdout, "	Texture Alignment: %lu\n", properties->textureAlignment);
	fprintf(stdout, "	Texture Pitch Alignment: %lu\n", properties->texturePitchAlignment);
	fprintf(stdout, "	Unified Addressing: %d\n", properties->unifiedAddressing);
}

error initializeDevs(void) {
	DPRINTF(1, "Initializing devices\n");
	for (natural curdev = 0; curdev < gpuCount; curdev++) {
		DPRINTF(2, "Initializing device %d\n", gpus[curdev].realdevice);
		CUDA_CHECK_RETURN(cudaSetDevice(gpus[curdev].realdevice));
		CUDA_CHECK_RETURN(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
		autotune(curdev);
		/*
		 * Create streams
		 */
		CUDA_CHECK_RETURN(cudaStreamCreate(&gpus[curdev].stream));
		DPRINTF(1, "Stream for device %lu = %p\n", curdev, gpus[curdev].stream);

		CUBLAS_CHECK_RETURN(cublasCreate_v2(&gpus[curdev].cublasHandle));
		CUBLAS_CHECK_RETURN(cublasSetPointerMode_v2(gpus[curdev].cublasHandle, CUBLAS_POINTER_MODE_HOST));
		CUBLAS_CHECK_RETURN(cublasSetStream_v2(gpus[curdev].cublasHandle, gpus[curdev].stream));
		CURAND_CHECK_RETURN(curandCreateGenerator(&gpus[curdev].curandHandle, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CHECK_RETURN(curandSetPseudoRandomGeneratorSeed(gpus[curdev].curandHandle, 1234L + curdev));
		DPRINTF(1, "Curand handle for device %lu = %p\n", curdev, gpus[curdev].curandHandle);
		/*
		 * Null variables
		 */
		gpus[curdev].data.ptr 		= NULL;
		gpus[curdev].sphere.ptr 	= NULL;
		gpus[curdev].eigd.ptr 		= NULL;
		gpus[curdev].eigv.ptr 		= NULL;
		gpus[curdev].means.ptr		= NULL;
		gpus[curdev].ldetS			= 0;

		cudaExtent nchannnchannnmmodels;
		nchannnchannnmmodels.width = currentConfig.nchannels * sizeof(real);
		nchannnchannnmmodels.height = currentConfig.nchannels;
		nchannnchannnmmodels.depth = currentConfig.nmmodels;

		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].rnxnwork, nchannnchannnmmodels))
		gpus[curdev].rnxnwork.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].rnxnwork);


		DPRINT_ALLOC(gpus[curdev].rnxnwork);
		gpus[curdev].rbswork.xsize = currentConfig.block_size;
		gpus[curdev].rbswork.ysize = 1;
		gpus[curdev].rbswork.pitch = currentConfig.block_size * sizeof(real);
		CUDA_CHECK_RETURN(cudaMalloc(&gpus[curdev].rbswork.ptr, gpus[curdev].rbswork.pitch))

		gpus[curdev].rbsxMwork.xsize = currentConfig.block_size;
		gpus[curdev].rbsxMwork.ysize = currentConfig.nmmodels;
		gpus[curdev].rbsxMwork.pitch = currentConfig.block_size * sizeof(real);

		CUDA_CHECK_RETURN(cudaMalloc(&gpus[curdev].rbsxMwork.ptr, gpus[curdev].rbsxMwork.pitch))

		gpus[curdev].rnwork.xsize = currentConfig.nchannels;
		gpus[curdev].rnwork.ysize = 1;
		CUDA_CHECK_RETURN(cudaMallocPitch(
					&gpus[curdev].rnwork.ptr,
					&gpus[curdev].rnwork.pitch,
					gpus[curdev].rnwork.xsize * sizeof(real),
					gpus[curdev].rnwork.ysize))
		DPRINT_ALLOC(gpus[curdev].rnwork);

		gpus[curdev].rnxbswork.xsize = currentConfig.nchannels;
		gpus[curdev].rnxbswork.ysize = currentConfig.block_size;
		CUDA_CHECK_RETURN(cudaMallocPitch(
					&gpus[curdev].rnxbswork.ptr,
					&gpus[curdev].rnxbswork.pitch,
					gpus[curdev].rnxbswork.xsize * sizeof(real),
					gpus[curdev].rnxbswork.ysize))
		DPRINT_ALLOC(gpus[curdev].rnxbswork);

		gpus[curdev].rnxbswork2.xsize = currentConfig.nchannels;
		gpus[curdev].rnxbswork2.ysize = currentConfig.block_size;
		CUDA_CHECK_RETURN(cudaMallocPitch(
					&gpus[curdev].rnxbswork2.ptr,
					&gpus[curdev].rnxbswork2.pitch,
					gpus[curdev].rnxbswork2.xsize * sizeof(real),
					gpus[curdev].rnxbswork2.ysize))
		DPRINT_ALLOC(gpus[curdev].rnxbswork2);

		gpus[curdev].rnxbswork3.xsize = currentConfig.nchannels;
		gpus[curdev].rnxbswork3.ysize = currentConfig.block_size;
		CUDA_CHECK_RETURN(cudaMallocPitch(
					&gpus[curdev].rnxbswork3.ptr,
					&gpus[curdev].rnxbswork3.pitch,
					gpus[curdev].rnxbswork3.xsize * sizeof(real),
					gpus[curdev].rnxbswork3.ysize))
		DPRINT_ALLOC(gpus[curdev].rnxbswork3);

		cudaExtent nchannbsnmmodels;
		nchannbsnmmodels.width = currentConfig.nchannels * sizeof(real);
		nchannbsnmmodels.height = currentConfig.block_size;
		nchannbsnmmodels.depth = currentConfig.nmmodels;
		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].b, nchannbsnmmodels))
		gpus[curdev].b.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].b);

		cudaExtent nchannbsnsdmnmmodels;
		nchannbsnsdmnmmodels.width = currentConfig.nchannels * sizeof(real);
		nchannbsnsdmnmmodels.height = currentConfig.block_size;
		nchannbsnsdmnmmodels.depth = currentConfig.nsdm * currentConfig.nmmodels;
		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].y, nchannbsnsdmnmmodels))
		gpus[curdev].y.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].y);


		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].Q, nchannbsnsdmnmmodels))
		gpus[curdev].Q.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].Q);

		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].z, nchannbsnsdmnmmodels))
		gpus[curdev].z.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].z);

		gpus[curdev].v.xsize = currentConfig.block_size;
		gpus[curdev].v.ysize = currentConfig.nmmodels;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&gpus[curdev].v.ptr,
			&gpus[curdev].v.pitch,
			gpus[curdev].v.xsize * sizeof(real),
			gpus[curdev].v.ysize))
		DPRINT_ALLOC(gpus[curdev].v)

		gpus[curdev].g.xsize = currentConfig.nchannels;
		gpus[curdev].g.ysize = currentConfig.block_size;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&gpus[curdev].g.ptr,
			&gpus[curdev].g.pitch,
			gpus[curdev].g.xsize * sizeof(real),
			gpus[curdev].g.ysize))
		DPRINT_ALLOC(gpus[curdev].g)

		cudaExtent nchannbsnsdm;
		nchannbsnsdm.width = currentConfig.nchannels * sizeof(real);
		nchannbsnsdm.height = currentConfig.block_size;
		nchannbsnsdm.depth = currentConfig.nsdm;
		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].u, nchannbsnsdm))
		gpus[curdev].u.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].u);


		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].fp, nchannbsnsdm))
		gpus[curdev].fp.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].fp);

		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].ufp, nchannbsnsdm))
		gpus[curdev].ufp.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].ufp);

		gpus[curdev].c.xsize = currentConfig.nchannels;
		gpus[curdev].c.ysize = currentConfig.nmmodels;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&gpus[curdev].c.ptr,
			&gpus[curdev].c.pitch,
			gpus[curdev].c.xsize * sizeof(real),
			gpus[curdev].c.ysize))
		DPRINT_ALLOC(gpus[curdev].c)


		cudaExtent nchannnsdmnmmodels;
		nchannnsdmnmmodels.width = currentConfig.nchannels * sizeof(real);
		nchannnsdmnmmodels.height = currentConfig.nsdm;
		nchannnsdmnmmodels.depth =  currentConfig.nmmodels;
		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].beta, nchannnsdmnmmodels))
		gpus[curdev].beta.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].beta);

		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].mu, nchannnsdmnmmodels))
		gpus[curdev].mu.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].mu);

		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].alpha, nchannnsdmnmmodels))
		gpus[curdev].alpha.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].alpha);

		CUDA_CHECK_RETURN(cudaMalloc3D(&gpus[curdev].rho, nchannnsdmnmmodels))
		gpus[curdev].rho.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(gpus[curdev].rho);



		gpus[curdev].Lt.xsize = currentConfig.block_size;
		gpus[curdev].Lt.ysize = currentConfig.nmmodels;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&gpus[curdev].Lt.ptr,
			&gpus[curdev].Lt.pitch,
			gpus[curdev].Lt.xsize * sizeof(real),
			gpus[curdev].Lt.ysize))

	}
	DPRINTF(1, "Initializing devices finished\n");
	return SUCCESS;
}
error finalizeDevs(void) {
	DPRINTF(1, "Finalizing devices\n");
	for (natural curdev = 0; curdev < gpuCount; curdev++) {
		CUDA_CHECK_RETURN(cudaSetDevice(gpus[curdev].realdevice));

		SAFE_CUDA_FREE(gpus[curdev].data);
		SAFE_CUDA_FREE(gpus[curdev].sphere);
		SAFE_CUDA_FREE(gpus[curdev].eigd);
		SAFE_CUDA_FREE(gpus[curdev].eigv);
		SAFE_CUDA_FREE(gpus[curdev].means);

		SAFE_CUDA_FREE(gpus[curdev].rnxnwork);
		SAFE_CUDA_FREE(gpus[curdev].rbswork);
		SAFE_CUDA_FREE(gpus[curdev].rnwork);
		SAFE_CUDA_FREE(gpus[curdev].rnxbswork);
		SAFE_CUDA_FREE(gpus[curdev].rnxbswork2);
		SAFE_CUDA_FREE(gpus[curdev].rnxbswork3);

		SAFE_CUDA_FREE(gpus[curdev].b);
		SAFE_CUDA_FREE(gpus[curdev].y);
		SAFE_CUDA_FREE(gpus[curdev].Q);
		SAFE_CUDA_FREE(gpus[curdev].z);

		SAFE_CUDA_FREE(gpus[curdev].v);

		SAFE_CUDA_FREE(gpus[curdev].g);
		SAFE_CUDA_FREE(gpus[curdev].u);
		SAFE_CUDA_FREE(gpus[curdev].fp);
		SAFE_CUDA_FREE(gpus[curdev].ufp);

		SAFE_CUDA_FREE(gpus[curdev].c)
		SAFE_CUDA_FREE(gpus[curdev].beta)
		SAFE_CUDA_FREE(gpus[curdev].mu)
		SAFE_CUDA_FREE(gpus[curdev].alpha)
		SAFE_CUDA_FREE(gpus[curdev].rho)

		SAFE_CUDA_FREE(gpus[curdev].Lt);

		CUDA_CHECK_RETURN(cudaStreamDestroy(gpus[curdev].stream));
		CURAND_CHECK_RETURN(curandDestroyGenerator(gpus[curdev].curandHandle));
		CUBLAS_CHECK_RETURN(cublasDestroy_v2(gpus[curdev].cublasHandle));

	}
	DPRINTF(1, "Finalizing devices finished\n");
	return SUCCESS;
}
