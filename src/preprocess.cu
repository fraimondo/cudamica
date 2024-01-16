/*
 * preprocess.cu
 *
 *  Created on: Aug 2, 2012
 *      Author: fraimondo
 */

#include <preprocess.h>
#include <helpers.h>
#include <tools.h>
#include <magma.h>
#include <kernels.h>

/*
 * Magic:
 * When needed, blocks may increase this variable.
 * When this reaches gridDim.x, then last block has finished.
 */
#ifndef BLOCKS_FINISHED
#define BLOCKS_FINISHED
__device__ unsigned int blocksFinished;
#endif

/*
 * Centers a dataset.
 *
 * Computes the mean of each channel and substracts it.
 *
 * set: the dataset to be centered
 */
error center() {
	DPRINTF(1, "Centering data\n");
	TICKTIMER_START(centering);
	CUDA_CHECK_RETURN(cudaSetDevice(gpus[0].realdevice));
	natural 		nchannels = currentConfig.nchannels;
	natural 		nsamples = currentConfig.nsamples;

	//DDEVWRITEMAT(gpus[0].stream, gpus[0].data, "datasets/pre-center.fdt");

	cudaPitchedPtr	dev_sums;
	dev_sums.xsize = nchannels;
	dev_sums.ysize = gpus[0].dimensions.c1cNs_11dim.nblocks.x;


	CUDA_CHECK_RETURN(cudaMallocPitch(
			&dev_sums.ptr,
			&dev_sums.pitch,
			dev_sums.xsize * sizeof(real),
			gpus[0].dimensions.c1cNs_11dim.nblocks.x));

	CUDA_KERNEL_CALL(getMean,
			gpus[0].dimensions.c1cNs_11dim.nblocks, gpus[0].dimensions.c1cNs_11dim.nthreads,
			0l, gpus[0].stream,
			(real *)gpus[0].data.ptr,
			nchannels,
			nsamples,
			gpus[0].dimensions.c1cNs_11dim.ndata,
			gpus[0].data.pitch/sizeof(real),
			(real *)dev_sums.ptr,
			dev_sums.pitch/sizeof(real)
		);

	CUDA_CHECK_RETURN(cudaStreamSynchronize(gpus[0].stream));

	CUDA_KERNEL_CALL(subMean,
			gpus[0].dimensions.c1c1s_12dim.nblocks, gpus[0].dimensions.c1c1s_12dim.nthreads,
			0l, gpus[0].stream,
			(real *)gpus[0].data.ptr,
			gpus[0].data.pitch/sizeof(real),
			(const real *)dev_sums.ptr
		);

	CUDA_CHECK_RETURN(cudaMemcpy(gpus[0].means.ptr, dev_sums.ptr, nchannels * sizeof(real), cudaMemcpyDeviceToDevice));
	//DDEVWRITEMAT(gpus[0].stream, dev_sums, "datasets/means.fdt");

	CUDA_CHECK_RETURN(cudaFree(dev_sums.ptr));
	TICKTIMER_END(centering);

	//DDEVWRITEMAT(gpus[0].stream, gpus[0].data, "datasets/centered.fdt");

	DPRINTF(1, "Centering dataset finished!\n");

	return SUCCESS;

}



error whiten() {
	DPRINTF(1,"Whitening dataset\n");
	TICKTIMER_START(whitening);
	cudaPitchedPtr	dev_sphere;
	cudaPitchedPtr	dev_eigv;

	//cudaPitchedPtr	dev_eigd;

	natural 		nchannels = currentConfig.nchannels;
	natural 		nsamples = currentConfig.nsamples;


	dev_sphere.xsize = nchannels;
	dev_sphere.ysize = nchannels;

	dev_eigv.xsize = nchannels;
	dev_eigv.ysize = nchannels;

	/* Sphere only on first device */
	CUDA_CHECK_RETURN(cudaSetDevice(gpus[0].realdevice));

	DPRINTF(2, "cudaMallocPitch %lu rows of %lu bytes for sphere matrix\n", dev_sphere.xsize, dev_sphere.ysize * sizeof(real));
	CUDA_CHECK_RETURN(cudaMallocPitch(&dev_sphere.ptr, &dev_sphere.pitch, dev_sphere.xsize * sizeof(real), dev_sphere.ysize));



	if (currentConfig.do_sphere == 1){
		/* Original = [Us, Ss, Vs] = svd(x*x'/N) */

		DPRINTF(2, "cudaMallocPitch %lu rows of %lu bytes for eigv matrix\n", dev_eigv.ysize, dev_eigv.xsize * sizeof(real));
		CUDA_CHECK_RETURN(cudaMallocPitch(&dev_eigv.ptr, &dev_eigv.pitch, dev_eigv.xsize * sizeof(real), dev_eigv.ysize));

		/*
		 * dev_sphere = data * data' / N;
		 */
		cublasHandle_t handle = gpus[0].cublasHandle;

		const real alpha = 1.0/(real)(nsamples);
		const real beta = 0.0;

		CUBLAS_CHECK_RETURN(
			cublasDsyrk_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
					nchannels, nsamples,
					&alpha, (real *)gpus[0].data.ptr, gpus[0].data.pitch/sizeof(real),
					&beta,  (real *)dev_sphere.ptr, dev_sphere.pitch/sizeof(real))
		);


		/* Compute eigenvalues and eigenvectors */
		int lwork = 1 + 6 * nchannels + 2 * nchannels * nchannels;
		int liwork = 3 + 5 * nchannels + 2;
		int info = 0;
		DPRINTF(2, "magma_dsyevd_gpu returned %d , lwork = %d, liwork = %d\n", info, lwork, liwork);

		cudaPitchedPtr 	host_eigd;
		host_eigd.xsize = nchannels;
		host_eigd.ysize = 1;
		host_eigd.pitch = nchannels * sizeof(real);

		cudaPitchedPtr 	host_wa;
		host_wa.xsize = nchannels;
		host_wa.ysize = nchannels;
		host_wa.pitch = nchannels * sizeof(real);

		cudaPitchedPtr 	host_ipiv;
		host_ipiv.xsize = nchannels;
		host_ipiv.ysize = 1;
		host_ipiv.pitch = nchannels * sizeof(int);

		CUDA_CHECK_RETURN(cudaHostAlloc(&host_eigd.ptr, host_eigd.xsize * sizeof(real) * host_eigd.ysize, cudaHostAllocDefault));
		CUDA_CHECK_RETURN(cudaHostAlloc(&host_wa.ptr, host_wa.xsize * sizeof(real) * host_wa.ysize, cudaHostAllocDefault));
		CUDA_CHECK_RETURN(cudaHostAlloc(&host_ipiv.ptr, host_ipiv.xsize * sizeof(int), cudaHostAllocDefault));


		real * host_work;
		int * host_iwork;



		CUDA_CHECK_RETURN(cudaHostAlloc(&host_work, lwork * sizeof(real), cudaHostAllocDefault));
		CUDA_CHECK_RETURN(cudaHostAlloc(&host_iwork, liwork * sizeof(int), cudaHostAllocDefault));

		//DDEVWRITEMAT(gpus[0].stream, dev_sphere, "datasets/pre-sphere.fdt");

		MAGMA_CHECK_RETURN(
			magma_dsyevd_gpu(
					MagmaVec,
					MagmaUpper,
					nchannels,
					(real *)dev_sphere.ptr,
					dev_sphere.pitch/sizeof(real),
					(real *)host_eigd.ptr,
					(real *)host_wa.ptr,
					host_wa.pitch/sizeof(real),
					(real *)host_work,
					lwork,
					(int *)host_iwork,
					liwork,
					&info)
		);

		DPRINTF(2, "magma_dsyevd_gpu returned %d\n", info);


		for (int i = 0; i < nchannels; i++) {
			CUBLAS_CHECK_RETURN(
				cublasDcopy_v2(handle,
					nchannels,
					(real*)((char*)dev_sphere.ptr + dev_sphere.pitch * i),
					1,
					(real*)dev_eigv.ptr + i,
					dev_eigv.pitch/sizeof(real)
			));
		}


		CUDA_CHECK_RETURN(cudaMemcpy2D(dev_sphere.ptr, dev_sphere.pitch, dev_eigv.ptr, dev_eigv.pitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));
		for (int i = 0; i < nchannels; i++) {
			((real*)host_eigd.ptr)[i] = 0.5 * sqrt(((real*)host_eigd.ptr)[i]);
			CUBLAS_CHECK_RETURN(
				cublasDscal_v2(handle,
					nchannels,
					&((real*)host_eigd.ptr)[i],
					(real*)dev_eigv.ptr + i,
					dev_eigv.pitch/sizeof(real)
			));
		}



		MAGMA_CHECK_RETURN(
			magma_dgesv_gpu(
					nchannels,
					nchannels,
					(real *)dev_eigv.ptr,
					dev_eigv.pitch/sizeof(real),
					(int *)host_ipiv.ptr,
					(real *)dev_sphere.ptr,
					dev_sphere.pitch/sizeof(real),
					&info)
		);



		/*
		 * Original: ldetS = -log(abs(det(sph)))
		 */
		CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
			gpus[0].rnxnwork.ptr,
			gpus[0].rnxnwork.pitch,
			dev_sphere.ptr,
			dev_sphere.pitch,
			dev_sphere.xsize * sizeof(real),
			dev_sphere.ysize,
			cudaMemcpyDeviceToDevice,
			gpus[0].stream
		));

		MAGMA_CHECK_RETURN(magma_dgetrf_gpu(
			gpus[0].rnxnwork.xsize,
			gpus[0].rnxnwork.ysize,
			(real*)gpus[0].rnxnwork.ptr,
			gpus[0].rnxnwork.pitch/sizeof(real),
			(int*)host_ipiv.ptr,
			&info
		));

		CUDA_KERNEL_CALL(getDiagonalMult,
			dim3(1, 1, 1),
			dim3(1, 1, 1),
			0l,
			gpus[0].stream,
			(real*)gpus[0].rnxnwork.ptr,
			gpus[0].rnxnwork.pitch/sizeof(real),
			gpus[0].rnxnwork.ysize
		);
		int sign = 1;
		for (int i = 0; i < nchannels; i++) {
			if (((int*)host_ipiv.ptr)[i] != i) {
				sign = -sign;
			}
		}
		//DDEVWRITEMAT(curmodelptr->dev_acopy[0], "datasets/alu.fdt");
		CUDA_CHECK_RETURN(cudaMemcpyAsync(
			&gpus[0].ldetS,
			(real*)gpus[0].rnxnwork.ptr,
			sizeof(real),
			cudaMemcpyDeviceToHost,
			gpus[0].stream
		));

		if (sign < 0) {
			gpus[0].ldetS = - gpus[0].ldetS;
		}
		CUDA_CHECK_RETURN(cudaFreeHost(host_wa.ptr));
		CUDA_CHECK_RETURN(cudaFreeHost(host_ipiv.ptr));
		CUDA_CHECK_RETURN(cudaFreeHost(host_work));
		CUDA_CHECK_RETURN(cudaFreeHost(host_iwork));

		DPRINTF(1, "Writing sphere matrix at %p (%lu x %lu) to %s\n", dev_sphere.ptr, dev_sphere.ysize, dev_sphere.xsize, currentConfig.spherefile);
		C_CHECK_RETURN(writeDevMatrix(dev_sphere, currentConfig.spherefile));
	} else if (currentConfig.do_sphere == 2) {
		/*
		 * Original: sphere = diag(1/sqrt(var(x)))
		 * Checked: Ok
		 */
		CUDA_CHECK_RETURN(cudaMemset2D(dev_sphere.ptr, dev_sphere.pitch, 0, dev_sphere.xsize * sizeof(real), dev_sphere.ysize));


		cudaPitchedPtr	dev_sums;
			dev_sums.xsize = nchannels;
			dev_sums.ysize = gpus[0].dimensions.c1cNs_11dim.nblocks.x;


		CUDA_CHECK_RETURN(cudaMallocPitch(
			&dev_sums.ptr,
			&dev_sums.pitch,
			dev_sums.xsize * sizeof(real),
			gpus[0].dimensions.c1cNs_11dim.nblocks.x));

		CUDA_KERNEL_CALL(getMean,
			gpus[0].dimensions.c1cNs_11dim.nblocks, gpus[0].dimensions.c1cNs_11dim.nthreads,
			0l, gpus[0].stream,
			(real *)gpus[0].data.ptr,
			nchannels,
			nsamples,
			gpus[0].dimensions.c1cNs_11dim.ndata,
			gpus[0].data.pitch/sizeof(real),
			(real *)dev_sums.ptr,
			dev_sums.pitch/sizeof(real)
		);

		CUDA_KERNEL_CALL(
			getvariance,
			dim3(1),
			dim3(nchannels),
			0l,
			gpus[0].stream,
			(real *)gpus[0].data.ptr, gpus[0].data.pitch/sizeof(real),
			nsamples,
			(real*)dev_sphere.ptr, dev_sphere.pitch/sizeof(real),
			(real*)dev_sums.ptr);

		CUDA_CHECK_RETURN(cudaStreamSynchronize(gpus[0].stream));
		DPRINTF(1, "Writing sphere matrix at %p (%lu x %lu) to %s\n", dev_sphere.ptr, dev_sphere.ysize, dev_sphere.xsize, currentConfig.spherefile);
		C_CHECK_RETURN(writeDevMatrix(dev_sphere, currentConfig.spherefile));
	} else {
		CUDA_KERNEL_CALL(
				eye,
				dim3(nchannels),
				dim3(nchannels),
				0l,
				gpus[0].stream,
				(real*)dev_sphere.ptr,
				dev_sphere.pitch/sizeof(real));
	}

	/* Determinant of diagonal matrix */
	if (currentConfig.do_sphere != 1) {
		CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
			gpus[0].rnxnwork.ptr,
			gpus[0].rnxnwork.pitch,
			dev_sphere.ptr,
			dev_sphere.pitch,
			dev_sphere.xsize * sizeof(real),
			dev_sphere.ysize,
			cudaMemcpyDeviceToDevice,
			gpus[0].stream
		));
		CUDA_KERNEL_CALL(getDiagonalMult,
			dim3(1, 1, 1),
			dim3(1, 1, 1),
			0l,
			gpus[0].stream,
			(real*)gpus[0].rnxnwork.ptr,
			gpus[0].rnxnwork.pitch/sizeof(real),
			gpus[0].rnxnwork.ysize
		);
		CUDA_CHECK_RETURN(cudaMemcpyAsync(
			&gpus[0].ldetS,
			(real*)gpus[0].rnxnwork.ptr,
			sizeof(real),
			cudaMemcpyDeviceToHost,
			gpus[0].stream
		));
	}

	CUDA_CHECK_RETURN(cudaStreamSynchronize(gpus[0].stream));
	gpus[0].ldetS = -log(abs(gpus[0].ldetS));
	DPRINTF(1, "ldets = %e\n", gpus[0].ldetS);

#ifdef ITERTEST
	C_CHECK_RETURN(writeValue(gpus[0].ldetS, "datasets/INPUT-ldets.fdt"));
#endif

	gpus[0].sphere.ptr = dev_sphere.ptr;
	gpus[0].sphere.pitch = dev_sphere.pitch;
	gpus[0].sphere.xsize = dev_sphere.xsize;
	gpus[0].sphere.ysize = dev_sphere.ysize;

	const real alpha = 1.0;
	const real beta = 0.0;

	cudaPitchedPtr newdata;
	newdata.xsize = nchannels;
	newdata.ysize = nsamples;
	CUDA_CHECK_RETURN(cudaMallocPitch(&newdata.ptr, &newdata.pitch, newdata.xsize * sizeof(real), newdata.ysize));
	CUDA_CHECK_RETURN(cudaMemcpy2D(newdata.ptr, newdata.pitch, gpus[0].data.ptr, gpus[0].data.pitch, newdata.xsize * sizeof(real), newdata.ysize, cudaMemcpyDeviceToDevice));



	CUBLAS_CHECK_RETURN(cublasDgemm_v2(
		gpus[0].cublasHandle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		nchannels,
		nsamples,
		nchannels,
		&alpha,
		(const real *)dev_sphere.ptr,
		dev_sphere.pitch/sizeof(real),
		(const real *)newdata.ptr,
		newdata.pitch/sizeof(real),
		&beta,
		(real *)gpus[0].data.ptr,
		gpus[0].data.pitch/sizeof(real)
	));
	//DDEVWRITEMAT(gpus[0].stream, gpus[0].data, "datasets/whitened.fdt");
	TICKTIMER_END(whitening);
	DPRINTF(1,"Whitening dataset finished\n");
	return SUCCESS;
}
