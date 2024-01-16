/*
 * amica.cu
 *
 *  Created on: Jul 23, 2012
 *      Author: fraimondo
 */

#include <tipos.h>
#include <amica.h>
#include <stdio.h>
#include <tools.h>
#include <config.h>
#include <preprocess.h>
#include <curand.h>
#include <mkl.h>

#include <magma.h>

#include <kernels.h>

model_block_t 		*blocks;				//blocks
natural				nblocks;				//number of blocks

/*
 * Calculates pinv for a given model host_pinva matrix and stores it in host_pinvv
 */
error pinv(model_t * model) {

	cudaPitchedPtr * a = &model->host_pinva;
	cudaPitchedPtr * u = &model->host_pinvu;
	cudaPitchedPtr * s = &model->host_pinvs;
	cudaPitchedPtr * sdiag = &model->host_pinvsdiag;
	cudaPitchedPtr * v = &model->host_pinvv;
	cudaPitchedPtr * superb = &model->host_pinvsuperb;


//	DWRITEMAT(*a,"datasets/pinva1.fdt");
	/*
	 * U S V = SVD(A)
	 */
	MKL_CHECK_RETURN(LAPACKE_dgesvd(
		LAPACK_COL_MAJOR,
		'A',
		'A',
		a->xsize,
		a->ysize,
		(real*)a->ptr,
		a->pitch/sizeof(real),
		(real*)s->ptr,
		(real*)u->ptr,
		u->pitch/sizeof(real),
		(real*)v->ptr,
		u->pitch/sizeof(real),
		(real*)superb->ptr
	));


//	DWRITEMAT(*u,"datasets/pinvu.fdt");
//	DWRITEMAT(*s,"datasets/pinvs.fdt");
//	DWRITEMAT(*v,"datasets/pinvv1.fdt");

	real * sptr = (real*)s->ptr;
	real * sdiagptr = (real*)sdiag->ptr;
	int elements = s->xsize;
	for (int i = 0; i < s->xsize; i++) {
		if (sptr[i] != 0) {
			sdiagptr[i + sdiag->xsize * i] = (real)1/sptr[i];
		} else {
			elements = i;
			break;
		}
	}

	if (elements != s->xsize) {
		memset(a->ptr, 0, a->ysize * a->pitch);
	}

	/*
	 * a = V * 1/s
	 */
	cblas_dgemm(
		CblasColMajor,
		CblasTrans,
		CblasNoTrans,
		v->xsize,
		sdiag->ysize,
		elements,
		1,
		(real*)v->ptr,
		v->pitch/sizeof(real),
		(real*)sdiag->ptr,
		sdiag->pitch/sizeof(real),
		0,
		(real*)a->ptr,
		a->pitch/sizeof(real)
	);
//	DWRITEMAT(*a,"datasets/pinva2.fdt");


	if (elements != s->xsize) {
		memset(v->ptr, 0, v->ysize * v->pitch);
	}

	/*
	 * V = a * U'
	 */
	cblas_dgemm(
		CblasColMajor,
		CblasNoTrans,
		CblasTrans,
		a->xsize,
		u->xsize,
		elements,
		1,
		(real*)a->ptr,
		a->pitch/sizeof(real),
		(real*)u->ptr,
		u->pitch/sizeof(real),
		0,
		(real*)v->ptr,
		v->pitch/sizeof(real)
	);


//	DWRITEMAT(*v,"datasets/pinvv2.fdt");
	return SUCCESS;
}

model_t mmodels[MAX_MODELS];


/* Initialize Variables for each model:
 * Each matrix is initialized in the first device for that model.
 */
error initializeModels() {
	DPRINTF(1, "Initializing models\n");
	natural devcount = gpuCount;
	natural nchannels = currentConfig.nchannels;
	natural nsamples = currentConfig.nsamples;
	natural nmmodels = currentConfig.nmmodels;
	natural nsdm = currentConfig.nsdm;
	nblocks = currentConfig.blocks_per_gpu_per_model * devcount;


	/*
	 * Assign each model a master device and alloc variables
	 */
	natural curdev = 0;
	for (natural curmodel = 0; curmodel < nmmodels; curmodel++){
		DPRINTF(2, "Initializing model %lu\n", curmodel);
		mmodels[curmodel].master_device = curdev;
		model_t *curmodelptr = &mmodels[curmodel];
		device_t* curdeviceptr = &gpus[curmodelptr->master_device];
		CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));

		/* Host variables initizalition */
		mmodels[curmodel].host_vsumsum = 0.0;


		// Alloc model variables

		mmodels[curmodel].dev_a.xsize = nchannels;
		mmodels[curmodel].dev_a.ysize = nchannels;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&mmodels[curmodel].dev_a.ptr,
			&mmodels[curmodel].dev_a.pitch,
			mmodels[curmodel].dev_a.xsize * sizeof(real),
			mmodels[curmodel].dev_a.ysize
		))

		mmodels[curmodel].dev_c.xsize = nchannels;
		mmodels[curmodel].dev_c.ysize = 1;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&mmodels[curmodel].dev_c.ptr,
			&mmodels[curmodel].dev_c.pitch,
			mmodels[curmodel].dev_c.xsize * sizeof(real),
			mmodels[curmodel].dev_c.ysize
		))
		DPRINT_ALLOC(mmodels[curmodel].dev_c);

		mmodels[curmodel].dev_beta.xsize = nchannels;
		mmodels[curmodel].dev_beta.ysize = nsdm;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&mmodels[curmodel].dev_beta.ptr,
			&mmodels[curmodel].dev_beta.pitch,
			mmodels[curmodel].dev_beta.xsize * sizeof(real),
			mmodels[curmodel].dev_beta.ysize
		))
		mmodels[curmodel].dev_mu.xsize = nchannels;
		mmodels[curmodel].dev_mu.ysize = nsdm;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&mmodels[curmodel].dev_mu.ptr,
			&mmodels[curmodel].dev_mu.pitch,
			mmodels[curmodel].dev_mu.xsize * sizeof(real),
			mmodels[curmodel].dev_mu.ysize
		))
		mmodels[curmodel].dev_alpha.xsize = nchannels;
		mmodels[curmodel].dev_alpha.ysize = nsdm;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&mmodels[curmodel].dev_alpha.ptr,
			&mmodels[curmodel].dev_alpha.pitch,
			mmodels[curmodel].dev_alpha.xsize * sizeof(real),
			mmodels[curmodel].dev_alpha.ysize
		))
		mmodels[curmodel].dev_rho.xsize = nchannels;
		mmodels[curmodel].dev_rho.ysize = nsdm;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&mmodels[curmodel].dev_rho.ptr,
			&mmodels[curmodel].dev_rho.pitch,
			mmodels[curmodel].dev_rho.xsize * sizeof(real),
			mmodels[curmodel].dev_rho.ysize
		))

		mmodels[curmodel].dev_ltall.xsize = nsamples;
		mmodels[curmodel].dev_ltall.ysize = 1;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&mmodels[curmodel].dev_ltall.ptr,
			&mmodels[curmodel].dev_ltall.pitch,
			mmodels[curmodel].dev_ltall.xsize * sizeof(real),
			mmodels[curmodel].dev_ltall.ysize
		))
		DPRINT_ALLOC(mmodels[curmodel].dev_ltall);


		mmodels[curmodel].host_pinva.xsize = nchannels;
		mmodels[curmodel].host_pinva.ysize = nchannels;
		mmodels[curmodel].host_pinva.pitch = nchannels * sizeof(real);
		CUDA_CHECK_RETURN(cudaMallocHost(&mmodels[curmodel].host_pinva.ptr,
				mmodels[curmodel].host_pinva.pitch * mmodels[curmodel].host_pinva.ysize))

		mmodels[curmodel].host_pinvu.xsize = nchannels;
		mmodels[curmodel].host_pinvu.ysize = nchannels;
		mmodels[curmodel].host_pinvu.pitch = nchannels * sizeof(real);
		CUDA_CHECK_RETURN(cudaMallocHost(&mmodels[curmodel].host_pinvu.ptr,
				mmodels[curmodel].host_pinvu.pitch * mmodels[curmodel].host_pinvu.ysize))

		mmodels[curmodel].host_pinvs.xsize = nchannels;
		mmodels[curmodel].host_pinvs.ysize = 1;
		mmodels[curmodel].host_pinvs.pitch = nchannels * sizeof(real);
		CUDA_CHECK_RETURN(cudaMallocHost(&mmodels[curmodel].host_pinvs.ptr,
				mmodels[curmodel].host_pinvs.pitch * mmodels[curmodel].host_pinvs.ysize))

		mmodels[curmodel].host_pinvsdiag.xsize = nchannels;
		mmodels[curmodel].host_pinvsdiag.ysize = nchannels;
		mmodels[curmodel].host_pinvsdiag.pitch = nchannels * sizeof(real);
		CUDA_CHECK_RETURN(cudaMallocHost(&mmodels[curmodel].host_pinvsdiag.ptr,
				mmodels[curmodel].host_pinvsdiag.pitch * mmodels[curmodel].host_pinvsdiag.ysize))

		mmodels[curmodel].host_pinvv.xsize = nchannels;
		mmodels[curmodel].host_pinvv.ysize = nchannels;
		mmodels[curmodel].host_pinvv.pitch = nchannels * sizeof(real);
		CUDA_CHECK_RETURN(cudaMallocHost(&mmodels[curmodel].host_pinvv.ptr,
				mmodels[curmodel].host_pinvv.pitch * mmodels[curmodel].host_pinvv.ysize))

		mmodels[curmodel].host_pinvsuperb.xsize = nchannels;
		mmodels[curmodel].host_pinvsuperb.ysize = nchannels;
		mmodels[curmodel].host_pinvsuperb.pitch = nchannels * sizeof(real);
		CUDA_CHECK_RETURN(cudaMallocHost(&mmodels[curmodel].host_pinvsuperb.ptr,
				mmodels[curmodel].host_pinvsuperb.pitch * mmodels[curmodel].host_pinvsuperb.ysize))

		mmodels[curmodel].host_ipiv.xsize = nchannels;
		mmodels[curmodel].host_ipiv.ysize = 1;
		mmodels[curmodel].host_ipiv.pitch = nchannels * sizeof(real);
		CUDA_CHECK_RETURN(cudaMallocHost(
			&mmodels[curmodel].host_ipiv.ptr,
			mmodels[curmodel].host_ipiv.xsize * sizeof(int)
		));

		/*
		 * Initialize model matrix
		 */
		CURAND_CHECK_RETURN(curandGenerateUniformDouble(
				curdeviceptr->curandHandle,
			(real*)mmodels[curmodel].dev_a.ptr,
			mmodels[curmodel].dev_a.pitch/sizeof(real) * mmodels[curmodel].dev_a.ysize
		));

		real sum = -0.5;
		real mul = 0.05;
		mpaddConstants<<<
				curdeviceptr->dimensions.c1c1c_11dim.nblocks,
				curdeviceptr->dimensions.c1c1c_11dim.nthreads,
				0l,
				curdeviceptr->stream>>>(
				(real*)mmodels[curmodel].dev_a.ptr,
				mmodels[curmodel].dev_a.pitch/sizeof(real),
				mul,
				sum
				);


		CUDA_KERNEL_CALL(addEye,
			curdeviceptr->dimensions.c1c1c_11dim.nblocks, curdeviceptr->dimensions.c1c1c_11dim.nthreads,
			0l, curdeviceptr->stream,
			(real*)mmodels[curmodel].dev_a.ptr,
			mmodels[curmodel].dev_a.pitch/sizeof(real)
		);

		CUDA_KERNEL_CALL(normalize,
				curdeviceptr->dimensions.c1c1c_11dim.nblocks, curdeviceptr->dimensions.c1c1c_11dim.nthreads,
			nchannels * sizeof(real), curdeviceptr->stream,
			(real *)curmodelptr->dev_a.ptr,
			curmodelptr->dev_a.pitch/sizeof(real)
		);

		CURAND_CHECK_RETURN(curandGenerateNormalDouble(
			curdeviceptr->curandHandle,
			(real*)curmodelptr->dev_c.ptr,
			curmodelptr->dev_c.pitch/sizeof(real),
			0,
			1
		));

#ifdef NORAND
		if (curmodel == 0) {
			DLOADMAT(curmodelptr->dev_a, "datasets/anorand.fdt");
			DLOADMAT(curmodelptr->dev_c, "datasets/cnorand.fdt");
		} else {
			DLOADMAT(curmodelptr->dev_a, "datasets/anorand1.fdt");
			DLOADMAT(curmodelptr->dev_c, "datasets/cnorand1.fdt");
		}
#endif


#ifdef ITERTEST
		{char nombre[80];
		CUDA_CHECK_RETURN(cudaStreamSynchronize(curdeviceptr->stream));
		printf("Writing input matrixes for model %d\n", curmodel);
		sprintf(nombre, "datasets/INPUT-A-m%lu.fdt", curmodel);
		C_CHECK_RETURN(writeDevMatrix(curmodelptr->dev_a, nombre));
		sprintf(nombre, "datasets/INPUT-C-m%lu.fdt", curmodel);
		C_CHECK_RETURN(writeDevMatrix(curmodelptr->dev_c, nombre));
		}

#endif

		curmodelptr->gm = 1.0/(real)nmmodels;

		real frac = (1.0)/(real)nsdm;

		constant2D<<<dim3(nsdm, 1, 1), 	dim3(nchannels, 1, 1), 0, curdeviceptr->stream>>>(
			(real *)curmodelptr->dev_alpha.ptr,
			curmodelptr->dev_alpha.pitch/sizeof(real),
			frac
		);

		if (currentConfig.fix_init) {

			for (natural m0 = 1; m0 <= nsdm; m0++) {
				frac = (real)m0 - 1.0 -(nsdm-1.0)/2.0;
				constant1D<<<dim3(1, 1, 1), 	dim3(nchannels, 1, 1), 0, curdeviceptr->stream>>>(
					OFFSET2D(curmodelptr->dev_mu, m0-1),
					frac
				);
			}
			//DDEVWRITEMAT(curdeviceptr->stream, curmodelptr->dev_mu, "datasets/mu.fdt")
			frac = 1.0;
			constant2D<<<dim3(nsdm, 1, 1), 	dim3(nchannels, 1, 1), 0, curdeviceptr->stream>>>(
				(real *)curmodelptr->dev_beta.ptr,
				curmodelptr->dev_beta.pitch/sizeof(real),
				frac
			);
		} else {
			if (nsdm > 1) {
				CURAND_CHECK_RETURN(curandGenerateNormalDouble(
					curdeviceptr->curandHandle,
					(real*)curmodelptr->dev_mu.ptr,
					curmodelptr->dev_mu.pitch/sizeof(real) * curmodelptr->dev_mu.ysize,
					0,
					0.1
				));
			} else {
				CUDA_CHECK_RETURN(cudaMemset(
					curmodelptr->dev_mu.ptr,
					0,
					curmodelptr->dev_mu.ysize * curmodelptr->dev_mu.pitch
				));

			}
			CURAND_CHECK_RETURN(curandGenerateNormalDouble(
				curdeviceptr->curandHandle,
				(real*)curmodelptr->dev_beta.ptr,
				curmodelptr->dev_beta.pitch/sizeof(real) * curmodelptr->dev_beta.ysize,
				1,
				0.1
			));
		}

		frac = currentConfig.rho0;
		constant2D<<<dim3(nsdm, 1, 1), 	dim3(nchannels, 1, 1), 0, curdeviceptr->stream>>>(
			(real *)curmodelptr->dev_rho.ptr,
			curmodelptr->dev_rho.pitch/sizeof(real),
			frac
		);

		curdev++;
		if (curdev == devcount) curdev = 0;
	}

	/*
	 * Initialize each block
	 * Important: block 0 starts on device 0, the order
	 * of block-device assignment is used during the processing.
	 * -- DO NOT CHANGE --
	 */
	curdev = 0;
	blocks = (model_block_t*)malloc(nblocks * sizeof(model_block_t));

	for (natural curblock = 0; curblock < nblocks; curblock++) {

		device_t* curdevice = &gpus[curdev];
		model_block_t * curblockptr = &blocks[curblock];
		blocks[curblock].device = curdev;
		blocks[curblock].stream = &gpus[curdev].stream;
		blocks[curblock].models = nmmodels;
		blocks[curblock].block_size = currentConfig.block_size;
		blocks[curblock].start = (curblock) * currentConfig.block_size;
		if (curblock == nblocks-1) {
			blocks[curblock].block_size = nsamples - blocks[curblock].start;
		}
		blocks[curblock].end = blocks[curblock].start + blocks[curblock].block_size;

		DPRINTF(2, "-- Block %lu st %lu end %lu size %lu dev %lu\n", curblock,
				blocks[curblock].start, blocks[curblock].end,
				blocks[curblock].block_size, blocks[curblock].device);
		CUDA_CHECK_RETURN(cudaSetDevice(curdevice->realdevice));

		// Alloc block variables
		cudaExtent nchannsdmnmmodels;
		nchannsdmnmmodels.width = currentConfig.nchannels * sizeof(real);
		nchannsdmnmmodels.height = currentConfig.nsdm;
		nchannsdmnmmodels.depth = currentConfig.nmmodels;
//		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dalpha_numer, nchannsdmnmmodels))
//		curblockptr->dev_dalpha_numer.xsize = currentConfig.nchannels;
//		DPRINT_ALLOC(curblockptr->dev_dalpha_numer);

		/*CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dalpha_denom, nchannsdmnmmodels))
		curblockptr->dev_dalpha_denom.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_dalpha_denom);*/

		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dmu_numer, nchannsdmnmmodels))
		curblockptr->dev_dmu_numer.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_dmu_numer);

		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dmu_denom, nchannsdmnmmodels))
		curblockptr->dev_dmu_denom.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_dmu_denom);

		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dbeta_numer, nchannsdmnmmodels))
		curblockptr->dev_dbeta_numer.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_dbeta_numer);

		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dbeta_denom, nchannsdmnmmodels))
		curblockptr->dev_dbeta_denom.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_dbeta_denom);

		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_drho_numer, nchannsdmnmmodels))
		curblockptr->dev_drho_numer.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_drho_numer);

//		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_drho_denom, nchannsdmnmmodels))
//		curblockptr->dev_drho_denom.xsize = currentConfig.nchannels;
//		DPRINT_ALLOC(curblockptr->dev_drho_denom);

		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dlambda_numer, nchannsdmnmmodels))
		curblockptr->dev_dlambda_numer.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_dlambda_numer);

//		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dlambda_denom, nchannsdmnmmodels))
//		curblockptr->dev_dlambda_denom.xsize = currentConfig.nchannels;
//		DPRINT_ALLOC(curblockptr->dev_dlambda_denom);

		curblockptr->dev_dsigma2_numer.xsize = nchannels;
		curblockptr->dev_dsigma2_numer.ysize = nsdm;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&curblockptr->dev_dsigma2_numer.ptr,
			&curblockptr->dev_dsigma2_numer.pitch,
			curblockptr->dev_dsigma2_numer.xsize * sizeof(real),
			curblockptr->dev_dsigma2_numer.ysize
		))
//		curblockptr->dev_dsigma2_denom.xsize = nchannels;
//		curblockptr->dev_dsigma2_denom.ysize = nsdm;
//		CUDA_CHECK_RETURN(cudaMallocPitch(
//			&curblockptr->dev_dsigma2_denom.ptr,
//			&curblockptr->dev_dsigma2_denom.pitch,
//			curblockptr->dev_dsigma2_denom.xsize * sizeof(real),
//			curblockptr->dev_dsigma2_denom.ysize
//		))

		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dkappa_numer, nchannsdmnmmodels))
		curblockptr->dev_dkappa_numer.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_dkappa_numer);

//		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_dkappa_denom, nchannsdmnmmodels))
//		curblockptr->dev_dkappa_denom.xsize = currentConfig.nchannels;
//		DPRINT_ALLOC(curblockptr->dev_dkappa_denom);


		cudaExtent nchannnchannnmmodels;
		nchannnchannnmmodels.width = currentConfig.nchannels * sizeof(real);
		nchannnchannnmmodels.height = currentConfig.nchannels;
		nchannnchannnmmodels.depth = currentConfig.nmmodels;
		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_phi, nchannnchannnmmodels))
		curblockptr->dev_phi.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_phi);

		curblockptr->dev_cnew.xsize = nchannels;
		curblockptr->dev_cnew.ysize = nmmodels;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&curblockptr->dev_cnew.ptr,
			&curblockptr->dev_cnew.pitch,
			curblockptr->dev_cnew.xsize * sizeof(real),
			curblockptr->dev_cnew.ysize
		))
		DPRINT_ALLOC(curblockptr->dev_cnew);

		curblockptr->dev_v.xsize = curblockptr->block_size;
		curblockptr->dev_v.ysize = nmmodels;
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&curblockptr->dev_v.ptr,
			&curblockptr->dev_v.pitch,
			curblockptr->dev_v.xsize * sizeof(real),
			curblockptr->dev_v.ysize
		))
		DPRINT_ALLOC(curblockptr->dev_v);


		CUDA_CHECK_RETURN(cudaMalloc3D(&curblockptr->dev_usum, nchannsdmnmmodels))
		curblockptr->dev_usum.xsize = currentConfig.nchannels;
		DPRINT_ALLOC(curblockptr->dev_usum);

		CUDA_CHECK_RETURN(cudaMallocHost(&curblockptr->host_usum.ptr, nchannels * nsdm * nmmodels * sizeof(real) ))
		curblockptr->host_usum.xsize = currentConfig.nchannels;
		curblockptr->host_usum.ysize = currentConfig.nsdm;
		curblockptr->host_usum.pitch = currentConfig.nchannels * sizeof(real);
		DPRINT_ALLOC(curblockptr->dev_usum);

		curblockptr->dev_vsum.xsize = nmmodels;
		curblockptr->dev_vsum.ysize = 1;
		curblockptr->dev_vsum.pitch = nmmodels*sizeof(real);
		CUDA_CHECK_RETURN(cudaMalloc(&curblockptr->dev_vsum.ptr, nmmodels*sizeof(real)));

		curblockptr->host_vsum.xsize = nmmodels;
		curblockptr->host_vsum.ysize = 1;
		curblockptr->host_vsum.pitch = nmmodels*sizeof(real);
		CUDA_CHECK_RETURN(cudaMallocHost(&curblockptr->host_vsum.ptr, nmmodels*sizeof(real)));

		curdev++;
		if (curdev == devcount) curdev = 0;
	}

//

	DPRINTF(1, "Initializing models finished\n");
	return SUCCESS;
}


/*
 * Free resources for each model
 */
error finalizeModels() {
	DPRINTF(1, "Finalizing models\n");
	natural nmmodels = currentConfig.nmmodels;

	for (natural curblock = 0; curblock < nblocks; curblock++) {
		model_block_t * curblockptr = &blocks[curblock];

//		SAFE_CUDA_FREE(curblockptr->dev_dalpha_numer)
		//SAFE_CUDA_FREE(curblockptr->dev_dalpha_denom)
		SAFE_CUDA_FREE(curblockptr->dev_dmu_numer)
		SAFE_CUDA_FREE(curblockptr->dev_dmu_denom)
		SAFE_CUDA_FREE(curblockptr->dev_dbeta_numer)
		SAFE_CUDA_FREE(curblockptr->dev_dbeta_denom)
		SAFE_CUDA_FREE(curblockptr->dev_drho_numer)
//		SAFE_CUDA_FREE(curblockptr->dev_drho_denom)

		SAFE_CUDA_FREE(curblockptr->dev_dlambda_numer)
//		SAFE_CUDA_FREE(curblockptr->dev_dlambda_denom)
		SAFE_CUDA_FREE(curblockptr->dev_dsigma2_numer)
		//SAFE_CUDA_FREE(curblockptr->dev_dsigma2_denom)
		SAFE_CUDA_FREE(curblockptr->dev_dkappa_numer)
//		SAFE_CUDA_FREE(curblockptr->dev_dkappa_denom)

		SAFE_CUDA_FREE(curblockptr->dev_phi)
		SAFE_CUDA_FREE(curblockptr->dev_cnew)
		SAFE_CUDA_FREE(curblockptr->dev_v)
		SAFE_CUDA_FREE(curblockptr->dev_usum)
		SAFE_FREE_HOST(curblockptr->host_usum)
		SAFE_CUDA_FREE(curblockptr->dev_vsum)
		SAFE_FREE_HOST(curblockptr->host_vsum)

	}

	for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
		model_t* curmodelptr = &mmodels[curmodel];

		SAFE_CUDA_FREE(curmodelptr->dev_a)
		SAFE_CUDA_FREE(curmodelptr->dev_c)
		SAFE_CUDA_FREE(curmodelptr->dev_beta)
		SAFE_CUDA_FREE(curmodelptr->dev_mu)
		SAFE_CUDA_FREE(curmodelptr->dev_alpha)
		SAFE_CUDA_FREE(curmodelptr->dev_rho)

		SAFE_CUDA_FREE(curmodelptr->dev_ltall)


		SAFE_FREE_HOST(curmodelptr->host_pinva)
		SAFE_FREE_HOST(curmodelptr->host_pinvu)
		SAFE_FREE_HOST(curmodelptr->host_pinvs)
		SAFE_FREE_HOST(curmodelptr->host_pinvsdiag)
		SAFE_FREE_HOST(curmodelptr->host_pinvv)
		SAFE_FREE_HOST(curmodelptr->host_pinvsuperb)

		SAFE_FREE_HOST(curmodelptr->host_ipiv)

	}

	SAFE_FREE(blocks);
	DPRINTF(1, "Finalizing models finished\n");
	return SUCCESS;
}

error runamica(void) {
	PRINT_LINE();
	fprintf(stdout, "Running AMICA\n");
	PRINT_LINE();

	DPRINTF(1, "Initializing magma\n");
	magma_init();
	DPRINTF(1, "Magma Ok\n");



	/*
	 * Variables for cublas
	 */
	real alpha;
	real beta;
	real value;




	/*
	 * Local parameter copy
	 */
	natural devcount = gpuCount;
	natural nchannels = currentConfig.nchannels;


	natural nsamples = currentConfig.nsamples;
	natural nmmodels = currentConfig.nmmodels;


	natural nsdm = currentConfig.nsdm;
	natural maxiter = currentConfig.maxiter;


	/*
	 * Host variables
	 */

	real * host_data;

	C_CHECK_RETURN(initializeDevs());

	/*
	 * Create and load host data
	 */
	CUDA_CHECK_RETURN(cudaMallocHost(&host_data, nchannels * nsamples * sizeof(real)));
	C_CHECK_RETURN(loadMatrix(host_data, nsamples, nchannels, currentConfig.datafile));


	TICKTIMER_START(dataload);
	CUDA_CHECK_RETURN(cudaSetDevice(gpus[0].realdevice));

	gpus[0].data.xsize = nchannels;
	gpus[0].data.ysize = nsamples;

	/*
	 * Alloc memory
	 */
	CUDA_CHECK_RETURN(cudaMallocPitch(
		&gpus[0].data.ptr,
		&gpus[0].data.pitch,
		gpus[0].data.xsize * sizeof(real),
		gpus[0].data.ysize
	));

	DPRINTF(1, "MallocPitch Device %d -> dev_data at %p with pitch %lu\n",
			gpus[0].realdevice, gpus[0].data.ptr, gpus[0].data.pitch);

	/*
	 * Copy data to device 0
	 */
	CUDA_CHECK_RETURN(cudaMemcpy2D(
		gpus[0].data.ptr, 					//dst
		gpus[0].data.pitch,		 			//dpitch
		host_data, 							//src
		nchannels * sizeof(real), 			//spitch
		gpus[0].data.xsize * sizeof(real),	//width
		gpus[0].data.ysize,					//height
		cudaMemcpyHostToDevice				//kind
	));

	CUDA_CHECK_RETURN(cudaFreeHost(host_data));

	TICKTIMER_END(dataload);

	gpus[0].means.xsize = nchannels;
	gpus[0].means.ysize = 1;
	gpus[0].means.pitch = nchannels * sizeof(real);
	CUDA_CHECK_RETURN(cudaMalloc(
		&gpus[0].means.ptr,
		nchannels * sizeof(real)
	));

	CUDA_CHECK_RETURN(cudaMemset(
		gpus[0].means.ptr,
		0,
		gpus[0].means.xsize * sizeof(real)
	));


	/* Pre process */
	center();
	whiten();


#ifdef NORAND
	DLOADMAT(gpus[0].data, "datasets/whitened.fdt", true);
	DLOADMAT(gpus[0].means, "datasets/means.fdt", true);
#endif


#ifdef ITERTEST
	printf("Writing data matrixes\n");
	CUDA_CHECK_RETURN(cudaStreamSynchronize(gpus[0].stream));
	C_CHECK_RETURN(writeDevMatrix(gpus[0].data, "datasets/INPUT-data.fdt"));
	C_CHECK_RETURN(writeDevMatrix(gpus[0].means, "datasets/INPUT-means.fdt"));
#endif

	/* Copy data to other devices */
	for (int curdev = 1; curdev < devcount; curdev++) {
		CUDA_CHECK_RETURN(cudaSetDevice(gpus[curdev].realdevice));

		gpus[curdev].data.xsize = nchannels;
		gpus[curdev].data.ysize = nsamples;

		/*
		 * Alloc memory
		 */
		CUDA_CHECK_RETURN(cudaMallocPitch(
			&gpus[curdev].data.ptr,
			&gpus[curdev].data.pitch,
			gpus[curdev].data.xsize * sizeof(real),
			gpus[curdev].data.ysize
		));

		DPRINTF(1, "MallocPitch Device %d -> dev_data at %p with pitch %lu\n",
				gpus[curdev].realdevice, gpus[curdev].data.ptr, gpus[curdev].data.pitch);

		/*
		 * Copy data
		 */
		CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
			gpus[curdev].data.ptr, 			//dst
			gpus[curdev].data.pitch, 		//dpitch
			gpus[0].data.ptr,				//src
			gpus[0].data.pitch, 			//spitch
			gpus[curdev].data.xsize * sizeof(real),		//width
			gpus[curdev].data.ysize,		//height
			cudaMemcpyDeviceToDevice,		//kind
			gpus[0].stream
		));
		gpus[curdev].means.xsize = nchannels;
		gpus[curdev].means.ysize = 1;
		gpus[curdev].means.pitch = nchannels * sizeof(real);
		CUDA_CHECK_RETURN(cudaMalloc(
			&gpus[curdev].means.ptr,
			nchannels * sizeof(real)
		));

		CUDA_CHECK_RETURN(cudaMemcpyAsync(gpus[curdev].means.ptr, gpus[0].means.ptr, nchannels * sizeof(real), cudaMemcpyDeviceToDevice, gpus[0].stream));

		gpus[curdev].ldetS = gpus[0].ldetS;

	}

	/*
	 * Initialize variables for each model
	 */
	C_CHECK_RETURN(initializeModels());

	/*
	 * Global amica variables
	 */

	real* host_ll = (real*)malloc((maxiter+1) * sizeof(real));
	real lrate = currentConfig.lrate0;
	real lratefact = currentConfig.lratefact;
	real lratemax = currentConfig.lratemax;
	natural numdec = currentConfig.numdec;
	natural maxdec = currentConfig.maxdec;


	/*
	 * Following code is AMICA.
	 * Only one model per device (yet) and one device per model (yet).
	 */
	DPRINTF(1, "Starting amica\n");
	model_t* curmodelptr = NULL;
	device_t * curdeviceptr = NULL;
	devicetune * curtune = NULL;
	model_block_t * curblockptr = NULL;

	bool end = false;
	fprintf(stdout, "Starting %d AMICA iterations\n", maxiter);
	for (natural iter=1; iter <= maxiter && !end; iter++) {
		host_ll[iter] = 0;
		TICKTIMER_START(iteration);
		SECTIMER_START(iteration);
		DPRINTF(2, "Starting iteration %lu\n", iter);

		for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
			curmodelptr =  &mmodels[curmodel];
			/*Reset variables */
			curmodelptr->host_vsumsum = 0;
		}

		/*
		 * Compute pinv(A) for each model
		 */
		for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
			curmodelptr =  &mmodels[curmodel];
			curdeviceptr = &gpus[curmodelptr->master_device];
			curtune = &curdeviceptr->dimensions;
			CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
				curmodelptr->host_pinva.ptr,
				curmodelptr->host_pinva.pitch,
				curmodelptr->dev_a.ptr,
				curmodelptr->dev_a.pitch,
				curmodelptr->dev_a.xsize * sizeof(real),
				curmodelptr->dev_a.ysize,
				cudaMemcpyDeviceToHost,
				curdeviceptr->stream
			));

			pinv(curmodelptr);
			DBUILD(3,
			if (curmodel == 0) {
				DWRITEMAT(curmodelptr->host_pinvv,"datasets/W0-0.fdt");
			} else {
				DWRITEMAT(curmodelptr->host_pinvv,"datasets/W0-1.fdt");
			})
		}


		/*
		 * Compute ldet(h)
		 */
		for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
			curmodelptr =  &mmodels[curmodel];
			curdeviceptr = &gpus[curmodelptr->master_device];
			curtune = &curdeviceptr->dimensions;
			CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));

			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
				OFFSET3D(curdeviceptr->W, curmodel),
				curdeviceptr->W.pitch,
				curmodelptr->host_pinvv.ptr,
				curmodelptr->host_pinvv.pitch,
				curmodelptr->host_pinvv.xsize * sizeof(real),
				curmodelptr->host_pinvv.ysize,
				cudaMemcpyHostToDevice,
				curdeviceptr->stream
			));

			MAGMA_CHECK_RETURN(magma_dgetrf_gpu(
				curdeviceptr->W.xsize,
				curdeviceptr->W.ysize,
				OFFSET3D(curdeviceptr->W, curmodel),
				curdeviceptr->W.pitch/sizeof(real),
				(int*)curmodelptr->host_ipiv.ptr,
				&curmodelptr->info

			));

			CUDA_KERNEL_CALL(getDiagonalMult,
				dim3(1, 1, 1),
				dim3(1, 1, 1),
				0l,
				curdeviceptr->stream,
				OFFSET3D(curdeviceptr->W, curmodel),
				curdeviceptr->W.pitch/sizeof(real),
				curdeviceptr->W.ysize
			);
//			int sign = 1;
//			for (int i = 0; i < nchannels; i++) {
//				if (((int*)curmodelptr->host_ipiv.ptr)[i] != i) {
//					sign = -sign;
//				}
//			}

			CUDA_CHECK_RETURN(cudaMemcpyAsync(
				&curmodelptr->ldet,
				OFFSET3D(curdeviceptr->W, curmodel),
				sizeof(real),
				cudaMemcpyDeviceToHost,
				curdeviceptr->stream
			));

//			if (sign < 0) {
//				curmodelptr->ldet = - curmodelptr->ldet;
//			}

			curmodelptr->ldet = log(abs(curmodelptr->ldet));

			DBUILD(1, CUDA_CHECK_RETURN(cudaDeviceSynchronize()));
			DPRINTF(1, "model %lu -> ldet = %.16f; gm = %.16f; ldetS = %.16f;\n", curmodel, curmodelptr->ldet, curmodelptr->gm, curdeviceptr->ldetS);

		}

		/*
		 * Start processing blocks
		 */
		for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
			curmodelptr =  &mmodels[curmodel];
			/*
			 * Copy c, beta, alpha, mu and rho to all the devices.
			 */
			{
				int curdev = 0;
				while (curdev < devcount) {
					curdeviceptr = &gpus[curdev];
					CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
					CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
						OFFSET2D(curdeviceptr->c, curmodel),
						curdeviceptr->c.pitch,
						curmodelptr->dev_c.ptr,
						curmodelptr->dev_c.pitch,
						curmodelptr->dev_c.xsize * sizeof(real),
						curmodelptr->dev_c.ysize,
						cudaMemcpyDeviceToDevice,
						curdeviceptr->stream
					));
					CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
						OFFSET3D(curdeviceptr->beta, curmodel),
						curdeviceptr->beta.pitch,
						curmodelptr->dev_beta.ptr,
						curmodelptr->dev_beta.pitch,
						curmodelptr->dev_beta.xsize * sizeof(real),
						curmodelptr->dev_beta.ysize,
						cudaMemcpyDeviceToDevice,
						curdeviceptr->stream
					));
					CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
						OFFSET3D(curdeviceptr->mu, curmodel),
						curdeviceptr->mu.pitch,
						curmodelptr->dev_mu.ptr,
						curmodelptr->dev_mu.pitch,
						curmodelptr->dev_mu.xsize * sizeof(real),
						curmodelptr->dev_mu.ysize,
						cudaMemcpyDeviceToDevice,
						curdeviceptr->stream
					));
					CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
						OFFSET3D(curdeviceptr->alpha, curmodel),
						curdeviceptr->alpha.pitch,
						curmodelptr->dev_alpha.ptr,
						curmodelptr->dev_alpha.pitch,
						curmodelptr->dev_alpha.xsize * sizeof(real),
						curmodelptr->dev_alpha.ysize,
						cudaMemcpyDeviceToDevice,
						curdeviceptr->stream
					));
					CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
						OFFSET3D(curdeviceptr->rho, curmodel),
						curdeviceptr->rho.pitch,
						curmodelptr->dev_rho.ptr,
						curmodelptr->dev_rho.pitch,
						curmodelptr->dev_rho.xsize * sizeof(real),
						curmodelptr->dev_rho.ysize,
						cudaMemcpyDeviceToDevice,
						curdeviceptr->stream
					));

					curdev++;
				}
			}
		}


		for (natural curblock = 0; curblock < nblocks; curblock++) {
			curblockptr = &blocks[curblock];
			curdeviceptr = &gpus[curblockptr->device];
			curtune = &curdeviceptr->dimensions;
			CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
			DPRINTF(2, "Iteration %lu - Block %lu\n", iter, curblock);
			for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
				curmodelptr = &mmodels[curmodel];
				/*
				 * Original: Lt(1:blocksize) = log(gm) + ldet + ldetS;
				 * Checked: Ok
				 * Difference: 0
				 */
				value = log(curmodelptr->gm) + curmodelptr->ldet + curdeviceptr->ldetS;
				DPRINTF(1,"block %lu, model %lu -> Lt value %.16f\n", curblock, curmodel, value);
				DBUILD(3,
					{char nombre[80];
					sprintf(nombre, "datasets/Ltvalue-b%dm%d.fdt", curblock, curmodel);
					C_CHECK_RETURN(writeValue(value, nombre));
					}
				)

				constant1D<<<curtune->vblocksize_11dim.nblocks,
						curtune->vblocksize_11dim.nthreads,
						0l,
						curdeviceptr->stream>>>(OFFSET2D(curdeviceptr->Lt,curmodel), value);
				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/Lt1-b%lu.fdt", curblock);
					DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->Lt, nombre)
				}
				})

				/*
				 * Original: b(1:blocksize) = W(i,:,h) * x(:,start:end) - W(i,:) * c(:)
				 * Checked: Ok (getb.m)
				 * Difference: -15
				 */
				CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
					OFFSET3D(curdeviceptr->W, curmodel),
					curdeviceptr->W.pitch,
					curmodelptr->host_pinvv.ptr,
					curmodelptr->host_pinvv.pitch,
					curmodelptr->host_pinvv.xsize * sizeof(real),
					curmodelptr->host_pinvv.ysize,
					cudaMemcpyHostToDevice,
					curdeviceptr->stream
				));
				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/W1-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->W, nmmodels, nombre)
				}
				})
				alpha = 1.0;
				beta = 0.0;
				CUBLAS_CHECK_RETURN(cublasDgemm_v2(
					curdeviceptr->cublasHandle,
					CUBLAS_OP_N,
					CUBLAS_OP_N,
					curdeviceptr->W.xsize,
					curblockptr->block_size,
					curdeviceptr->data.xsize,
					&alpha,
					OFFSET3D(curdeviceptr->W, curmodel),
					curdeviceptr->W.pitch/sizeof(real),
					OFFSET2D(curdeviceptr->data, curblockptr->start),
					curdeviceptr->data.pitch/sizeof(real),
					&beta,
					OFFSET3D(curdeviceptr->b, curmodel),
					curdeviceptr->b.pitch/sizeof(real)
				));
				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/c-b%lu.fdt", curblock);
					DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->c,  nombre)
				}
				})

				CUBLAS_CHECK_RETURN(cublasDgemv_v2(
					curdeviceptr->cublasHandle,
					CUBLAS_OP_N,
					curdeviceptr->W.ysize,
					curdeviceptr->W.xsize,
					&alpha,
					OFFSET3D(curdeviceptr->W,curmodel),
					curdeviceptr->W.pitch/sizeof(real),
					OFFSET2D(curdeviceptr->c,curmodel),
					1,
					&beta,
					(real*)curdeviceptr->rnwork.ptr,
					1
				));

				CUDA_KERNEL_CALL(
					substract,
					curtune->c1c1bs_12dim.nblocks,
					curtune->c1c1bs_12dim.nthreads,
					nchannels * sizeof(real),
					curdeviceptr->stream,
					OFFSET3D(curdeviceptr->b,curmodel),
					curdeviceptr->b.pitch/sizeof(real),
					(real*)curdeviceptr->rnwork.ptr
				);
				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/b2-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->b, nmmodels, nombre)
				}
				})

				/*
				 * Original: y(i,1:blksize,j) = sqrt(beta(i,j)) * b(i,1:blksize) - mu(i,j)
				 * Checked: Ok (betaminusmu.m)
				 * Difference: -15
				 */
				CUDA_KERNEL_CALL(
					betabyxminusmu,
					curtune->c1c1snsdm_22dim.nblocks,
					curtune->c1c1snsdm_22dim.nthreads,
					nchannels * sizeof(real) * 2, 	//one for beta, one for mu
					curdeviceptr->stream,
					OFFSET4D(curdeviceptr->y, curmodel, nsdm),
					curdeviceptr->y.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->b,curmodel),
					curdeviceptr->b.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->beta,curmodel),
					curdeviceptr->beta.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->mu,curmodel),
					curdeviceptr->mu.pitch/sizeof(real),
					curblockptr->block_size
				);

				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/y-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->y, nsdm * nmmodels, nombre)
					sprintf(nombre, "datasets/alpha-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->alpha, nmmodels, nombre)
					sprintf(nombre, "datasets/mu-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->mu, nmmodels, nombre)
					sprintf(nombre, "datasets/beta-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->beta, nmmodels, nombre)
				}
				})

				/*
				 * Original: Q(i,1:blocksize,j) = log(alpha(i,j)) + 0.5*log(beta(i,j)) + feval('logpfun', y(i,1:blocksize,j),rho(i,j));
				 * Checked: Ok (logalogblogpfun.m)
				 * Difference: -14
				 */
				CUDA_KERNEL_CALL(
					logalogblogpfun,
					curtune->c1c1snsdm_22dim.nblocks,
					curtune->c1c1snsdm_22dim.nthreads,
					nchannels * sizeof(real) * 2, 	//one for logalobblogpfun, one for rho
					curdeviceptr->stream,
					OFFSET4D(curdeviceptr->Q, curmodel, nsdm), curdeviceptr->Q.pitch/sizeof(real),
					OFFSET4D(curdeviceptr->y, curmodel, nsdm), curdeviceptr->y.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->alpha,curmodel), curdeviceptr->alpha.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->beta,curmodel), curdeviceptr->beta.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->rho,curmodel), curdeviceptr->rho.pitch/sizeof(real),
					curblockptr->block_size
				);
				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/Q-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->Q, nsdm * nmmodels, nombre)
					sprintf(nombre, "datasets/rho-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->rho, nmmodels, nombre)
				}
				})


				/*
				 * Original: amica13.m => Lines 243-253
				 * Checked: Ok (getltz)
				 * Difference: -15 (z) -13 (Lt)
				 */
				CUDA_KERNEL_CALL(
					updateltz,
					curtune->c1c1snsdm_12dim.nblocks,
					curtune->c1c1snsdm_12dim.nthreads,
					(nsdm+1) * nchannels * sizeof(real),
					curdeviceptr->stream,
					OFFSET4D(curdeviceptr->Q, curmodel, nsdm), curdeviceptr->Q.pitch/sizeof(real),
					OFFSET2D(curdeviceptr->Lt, curmodel), curdeviceptr->Lt.pitch/sizeof(real),
					OFFSET4D(curdeviceptr->z, curmodel, nsdm), curdeviceptr->z.pitch/sizeof(real)
				);

				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/z-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->z, nsdm * nmmodels, nombre)
					sprintf(nombre, "datasets/Lt2-b%lu.fdt", curblock);
					DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->Lt, nombre)
				}
				})

				/* Copy to Ltall */
				CUDA_CHECK_RETURN(cudaMemcpyAsync(
					OFFSET1D(curmodelptr->dev_ltall, curblockptr->start),
					OFFSET2D(curdeviceptr->Lt, curmodel),
					curblockptr->block_size * sizeof(real),
					cudaMemcpyDeviceToDevice,
					curdeviceptr->stream
				));
			}

			/* Update Lt
			 * Checked: OK (only v in getvll.m), OK (LL manually in getvll.m)
			 * Difference: -16 (v)
			 */
			CUDA_KERNEL_CALL(
				updatell,
				curtune->blocksize_12dim.nblocks,
				curtune->blocksize_12dim.nthreads,
				(nmmodels+1) * curtune->blocksize_12dim.nthreads.x * sizeof(real),
				curdeviceptr->stream,
				(real*)curdeviceptr->Lt.ptr, curdeviceptr->Lt.pitch/sizeof(real),
				(real*)curdeviceptr->v.ptr, curdeviceptr->v.pitch/sizeof(real),
				(real*)curdeviceptr->rbswork.ptr
			);
			DBUILD(3,
			{char nombre[80];
			sprintf(nombre, "datasets/Ll-b%lu.fdt", curblock);
			DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->rbswork, nombre)
			sprintf(nombre, "datasets/v-b%lu.fdt", curblock);
			DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->v, nombre)
			sprintf(nombre, "datasets/P-b%lu.fdt", curblock);
			DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->rbswork, nombre)
			})

			CUDA_CHECK_RETURN(cudaMemcpyAsync(
				&curblockptr->ll,
				(real*)curdeviceptr->rbswork.ptr,
				sizeof(real),
				cudaMemcpyDeviceToHost,
				curdeviceptr->stream
			));
			DBUILD(3,
			{
				CUDA_CHECK_RETURN(cudaStreamSynchronize(curdeviceptr->stream));
				DPRINTF(1, "block %lu -> LL value %.16f\n", curblock, curblockptr->ll);
			})

			/* Copy v to each block to store them till the end */
			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
				curblockptr->dev_v.ptr,
				curblockptr->dev_v.pitch,
				curdeviceptr->v.ptr,
				curdeviceptr->v.pitch,
				curdeviceptr->v.xsize * sizeof(real),
				curdeviceptr->v.ysize,
				cudaMemcpyDeviceToDevice,
				curdeviceptr->stream
			));

			for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
				curmodelptr = &mmodels[curmodel];


				CUDA_CHECK_RETURN(cudaMemset2DAsync(
					curdeviceptr->g.ptr,
					curdeviceptr->g.pitch,
					0,
					curdeviceptr->g.xsize * sizeof(real),
					curdeviceptr->g.ysize,
					curdeviceptr->stream
				));

				/* Update cnew
				 *
				 * Checked: Ok
				 */
				alpha = 1.0;
				beta = 0.0;
				CUBLAS_CHECK_RETURN(cublasDgemv_v2(
					curdeviceptr->cublasHandle,
					CUBLAS_OP_N,
					curdeviceptr->data.xsize,
					curblockptr->block_size,
					&alpha,
					OFFSET2D(curdeviceptr->data, curblockptr->start),
					curdeviceptr->data.pitch/sizeof(real),
					OFFSET2D(curdeviceptr->v, curmodel), 1,
					&beta,
					OFFSET2D(curblockptr->dev_cnew, curmodel),
					1
				));
				/*
				 * dsigma2_numer = sum(v.*b.^2)
				 *
				 * Checked: Ok
				 * Difference: 0 (pow) -10 (dsigma2_numer)
				 *
				 */
				CUDA_KERNEL_CALL(
					pow2,
					curtune->c1c1bs_12dim.nblocks,
					curtune->c1c1bs_12dim.nthreads,
					0,
					curdeviceptr->stream,
					OFFSET3D(curdeviceptr->b, curmodel), curdeviceptr->b.pitch/sizeof(real),
					(real*)curdeviceptr->rnxbswork.ptr, curdeviceptr->rnxbswork.pitch/sizeof(real)
				);
				alpha = 1.0;
				beta = 0.0;
				CUBLAS_CHECK_RETURN(cublasDgemv_v2(
					curdeviceptr->cublasHandle,
					CUBLAS_OP_N,
					nchannels,
					curblockptr->block_size,
					&alpha,
					(real*)curdeviceptr->rnxbswork.ptr,
					curdeviceptr->rnxbswork.pitch/sizeof(real),
					OFFSET2D(curdeviceptr->v, curmodel), 1,
					&beta,
					OFFSET2D(curblockptr->dev_dsigma2_numer, curmodel),
					1
				));
				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/bpow2-b%lu.fdt", curblock);
					DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->rnxbswork, nombre)
					sprintf(nombre, "datasets/s2_numer-b%lu.fdt", curblock);
					DDEVWRITEMAT(curdeviceptr->stream, curblockptr->dev_dsigma2_numer, nombre)
				}
				sprintf(nombre, "datasets/cnew-b%lu-m%lu.fdt", curblock, curmodel);
				DDEVWRITEMAT(curdeviceptr->stream, curblockptr->dev_cnew, nombre)
				})
				/*
				 * u = v .* z
				 * Checked: Ok (computeu.m)
				 * Difference: 0
				 *
				 */
				CUDA_KERNEL_CALL(
					computeu,
					curtune->c1c1snsdm_22dim.nblocks,
					curtune->c1c1snsdm_22dim.nthreads,
					curtune->c1c1snsdm_22dim.nthreads.y * sizeof(real),
					curdeviceptr->stream,
					OFFSET2D(curdeviceptr->v, curmodel),
					OFFSET4D(curdeviceptr->z, curmodel, nsdm), curdeviceptr->z.pitch/sizeof(real),
					(real*)curdeviceptr->u.ptr, curdeviceptr->u.pitch/sizeof(real)
				);
				DBUILD(3,
				{char nombre[80];
				sprintf(nombre, "datasets/u-b%lum%lu.fdt", curblock, curmodel);
				DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->u, nsdm, nombre)
				})


				/*
				 * usum = sum(u)
				 * Checked: Ok (computeu.m)
				 * Difference: -12
				 *
				 */
				CUDA_KERNEL_CALL(
					getusum,
					curtune->c1cNbs_21dim.nblocks,
					curtune->c1cNbs_21dim.nthreads,
					0,
					curdeviceptr->stream,
					(real*)curdeviceptr->u.ptr, curdeviceptr->u.pitch/sizeof(real),
					(real*)curdeviceptr->usumwork.ptr, curdeviceptr->usumwork.pitch/sizeof(real),
					OFFSET3D(curblockptr->dev_usum, curmodel), curblockptr->dev_usum.pitch/sizeof(real),
					curblockptr->block_size, curtune->c1cNbs_21dim.ndata
				);
				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/usum-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_usum, nmmodels, nombre)
				}
				})

				/*
				 * fp = ffun(y,rho)
				 * ufp = u * fp
				 * Checked: Ok (getfp.m)
				 * Difference: -18 (fp) -18 (ufp)
				 *
				 */
				CUDA_KERNEL_CALL(
					computefp,
					curtune->c1c1snsdm_22dim.nblocks,
					curtune->c1c1snsdm_22dim.nthreads,
					0,
					curdeviceptr->stream,
					OFFSET4D(curdeviceptr->y, curmodel, nsdm), curdeviceptr->y.pitch/sizeof(real),
					(real*)curdeviceptr->u.ptr, curdeviceptr->u.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->rho, curmodel), curdeviceptr->rho.pitch/sizeof(real),
					(real*)curdeviceptr->fp.ptr, curdeviceptr->fp.pitch/sizeof(real),
					(real*)curdeviceptr->ufp.ptr, curdeviceptr->ufp.pitch/sizeof(real)
				);
				DBUILD(3,
				{
				char nombre[80];
				sprintf(nombre, "datasets/ufp-b%lum%lu.fdt", curblock, curmodel);
				DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->ufp,nsdm, nombre)
				sprintf(nombre, "datasets/fp-b%lum%lu.fdt", curblock, curmodel);
				DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->fp,nsdm, nombre)

				});


				for (natural cursdm = 0; cursdm < nsdm; cursdm++) {
					/*
					 * g = g + sqrt(beta) * ufp
					 * Checked: Ok (getg.m)
					 * Difference: 0
					 */
					CUDA_KERNEL_CALL(
						computeg,
						curtune->c1c1bs_12dim.nblocks,
						curtune->c1c1bs_12dim.nthreads,
						curtune->c1c1bs_12dim.nthreads.x * sizeof(real),
						curdeviceptr->stream,
						OFFSET3D(curdeviceptr->ufp, cursdm), curdeviceptr->ufp.pitch/sizeof(real),
						OFFSET3D1D(curdeviceptr->beta, cursdm, curmodel),
						(real*)curdeviceptr->g.ptr, curdeviceptr->g.pitch/sizeof(real)
					);
					/*
					 * kappa_numer = kappa_numer + beta * sum (ufp * fp)
					 * Checked: Ok (kappa_numer.m)
					 * Difference: -11
					 */
					CUDA_KERNEL_CALL(
						getdkappanumer,
						curtune->c1cNbs_11dim.nblocks,
						curtune->c1cNbs_11dim.nthreads,
						0,
						curdeviceptr->stream,
						OFFSET3D(curdeviceptr->ufp, cursdm), curdeviceptr->ufp.pitch/sizeof(real),
						OFFSET3D(curdeviceptr->fp, cursdm), curdeviceptr->fp.pitch/sizeof(real),
						OFFSET3D1D(curdeviceptr->beta, cursdm, curmodel),
						(real*)curdeviceptr->rnxbswork.ptr, curdeviceptr->rnxbswork.pitch/sizeof(real),
						OFFSET3D1D(curblockptr->dev_dkappa_numer, cursdm, curmodel),
						curblockptr->block_size, curtune->c1cNbs_11dim.ndata
					);


					/*
					 * lambda_numer = lambda_numer + sum(u * ((fp * y) -1) ^2)
					 * Checked: Ok (lambda_numer.m)
					 * Difference: -10
					 */
					CUDA_KERNEL_CALL(
						getdlambdanumer,
						curtune->c1cNbs_11dim.nblocks,
						curtune->c1cNbs_11dim.nthreads,
						0,
						curdeviceptr->stream,
						OFFSET3D(curdeviceptr->u, cursdm), curdeviceptr->u.pitch/sizeof(real),
						OFFSET3D(curdeviceptr->fp, cursdm), curdeviceptr->fp.pitch/sizeof(real),
						OFFSET4D2D(curdeviceptr->y, cursdm, curmodel, nmmodels), curdeviceptr->y.pitch/sizeof(real),
						(real*)curdeviceptr->rnxbswork.ptr, curdeviceptr->rnxbswork.pitch/sizeof(real),
						OFFSET3D1D(curblockptr->dev_dlambda_numer, cursdm, curmodel),
						curblockptr->block_size, curtune->c1cNbs_11dim.ndata
					);
					/*
					 * Amica13.m lines 302-317
					 * Checked: Ok (betamu.m)
					 * Difference: rho <= 2 --> -11 (mu_*, beta_denom) 0 (beta_numer) | rho > 2 --> -11 (mu_numer) -9 (mu_denom) -10 (beta_denom) 0 (beta_numer)
					 */
					CUDA_KERNEL_CALL(
						getbetamu,
						curtune->c1cNbs_11dim.nblocks,
						curtune->c1cNbs_11dim.nthreads,
						0,
						curdeviceptr->stream,
						OFFSET3D(curdeviceptr->u, cursdm), curdeviceptr->u.pitch/sizeof(real),
						OFFSET3D(curdeviceptr->fp, cursdm), curdeviceptr->fp.pitch/sizeof(real),
						OFFSET3D(curdeviceptr->ufp, cursdm), curdeviceptr->ufp.pitch/sizeof(real),
						OFFSET4D2D(curdeviceptr->y, cursdm, curmodel, nmmodels), curdeviceptr->y.pitch/sizeof(real),
						(real*)curdeviceptr->rnxbswork.ptr, curdeviceptr->rnxbswork.pitch/sizeof(real),
						(real*)curdeviceptr->rnxbswork2.ptr, curdeviceptr->rnxbswork2.pitch/sizeof(real),
						(real*)curdeviceptr->rnxbswork3.ptr, curdeviceptr->rnxbswork3.pitch/sizeof(real),
						OFFSET3D1D(curblockptr->dev_dmu_numer, cursdm, curmodel),
						OFFSET3D1D(curblockptr->dev_dmu_denom, cursdm, curmodel),
						OFFSET3D1D(curblockptr->dev_dbeta_denom, cursdm, curmodel),
//						OFFSET3D1D(curblockptr->dev_dbeta_numer, cursdm, curmodel),
//						OFFSET2D(curblockptr->dev_usum, cursdm),
						OFFSET3D1D(curdeviceptr->rho, cursdm, curmodel),
						OFFSET3D1D(curdeviceptr->beta, cursdm, curmodel),
						curblockptr->block_size, curtune->c1cNbs_11dim.ndata
					);

					CUDA_KERNEL_CALL(
						getbetanumer,
						curtune->vchannels_11dim.nblocks,
						curtune->vchannels_11dim.nthreads,
						0,
						curdeviceptr->stream,
						OFFSET3D1D(curblockptr->dev_dbeta_numer, cursdm, curmodel),
						OFFSET3D1D(curblockptr->dev_usum, cursdm, curmodel),
						OFFSET3D1D(curdeviceptr->rho, cursdm, curmodel)
					);

					if (currentConfig.update_rho) {
						/*
						 * drho_numer = sum(u * abs(y)^rho * log(abs(y)^rho))
						 * Checked: Ok
						 * Difference: -11
						 */
						CUDA_KERNEL_CALL(
							getdrhonumer,
							curtune->c1cNbs_11dim.nblocks,
							curtune->c1cNbs_11dim.nthreads,
							0,
							curdeviceptr->stream,
							OFFSET3D(curdeviceptr->u, cursdm), curdeviceptr->u.pitch/sizeof(real),
							OFFSET4D2D(curdeviceptr->y, cursdm, curmodel, nmmodels), curdeviceptr->y.pitch/sizeof(real),
							OFFSET3D1D(curdeviceptr->rho, cursdm, curmodel),
							(real*)curdeviceptr->rnxbswork.ptr, curdeviceptr->rnxbswork.pitch/sizeof(real),
							OFFSET3D1D(curblockptr->dev_drho_numer, cursdm, curmodel),
							curblockptr->block_size, curtune->c1cNbs_11dim.ndata
						);
					}
				}

				/*
				 * phi = g * b'
				 * Checked: Ok
				 * Difference: -11
				 */
				alpha = 1.0;
				beta = 0.0;
				CUBLAS_CHECK_RETURN(cublasDgemm_v2(
					curdeviceptr->cublasHandle,
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					curdeviceptr->g.xsize,
					curdeviceptr->b.xsize,
					curblockptr->block_size,
					&alpha,
					(real*)curdeviceptr->g.ptr,
					curdeviceptr->g.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->b, curmodel),
					curdeviceptr->b.pitch/sizeof(real),
					&beta,
					OFFSET3D(curblockptr->dev_phi, curmodel),
					curblockptr->dev_phi.pitch/sizeof(real)
				));

				DBUILD(3,
				{char nombre[80];
				sprintf(nombre, "datasets/g-b%lum%lu.fdt", curblock, curmodel);
				DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->g, nombre)

				});
				DBUILD(3,
				{char nombre[80];
				if (curmodel == (nmmodels-1)) {
					sprintf(nombre, "datasets/kappa_numer-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dkappa_numer,nmmodels, nombre)
					sprintf(nombre, "datasets/lambda_numer-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dlambda_numer,nmmodels, nombre)
					sprintf(nombre, "datasets/mu_numer-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dmu_numer,nmmodels, nombre)
					sprintf(nombre, "datasets/mu_denom-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dmu_denom,nmmodels, nombre)
					sprintf(nombre, "datasets/beta_numer-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dbeta_numer,nmmodels, nombre)
					sprintf(nombre, "datasets/beta_denom-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dbeta_denom,nmmodels, nombre)
					sprintf(nombre, "datasets/rho_numer-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_drho_numer,nmmodels, nombre)
					sprintf(nombre, "datasets/phi-b%lu.fdt", curblock);
					DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_phi,nmmodels, nombre)
				}
				})


				/*
				 * sum v
				 */
				CUBLAS_CHECK_RETURN(cublasSetPointerMode_v2(curdeviceptr->cublasHandle, CUBLAS_POINTER_MODE_DEVICE));
				CUBLAS_CHECK_RETURN(
					cublasDasum_v2(
					curdeviceptr->cublasHandle,
					curblockptr->block_size,
					OFFSET2D(curblockptr->dev_v, curmodel),
					1,
					OFFSET1D(curblockptr->dev_vsum, curmodel*sizeof(real))
				));
				CUDA_CHECK_RETURN(cudaMemcpyAsync(
					OFFSET1D(curblockptr->host_vsum, curmodel*sizeof(real)),
					OFFSET1D(curblockptr->dev_vsum, curmodel*sizeof(real)),
					sizeof(real),
					cudaMemcpyDeviceToHost,
					curdeviceptr->stream
				));
				CUBLAS_CHECK_RETURN(cublasSetPointerMode_v2(curdeviceptr->cublasHandle, CUBLAS_POINTER_MODE_HOST));
//				curmodelptr->host_vsumsum += curmodelptr->host_vsum;
//
//
//				CUDA_CHECK_RETURN(cudaStreamSynchronize(curdeviceptr->stream));
//				DPRINTF(1, "vsum %.16f vsumsum %.16f\n", curmodelptr->host_vsum, curmodelptr->host_vsumsum)
			}

			host_ll[iter] += curblockptr->ll;


		}

		/*
		 * Sum every numer/denom.
		 *
		 * For each device, reduce it to the lowest block in it
		 */
		natural sblock = 0;
		for (natural curblock = devcount; curblock < nblocks; curblock++) {
			sblock = curblock % devcount;
			curblockptr = &blocks[curblock];
			curdeviceptr = &gpus[curblockptr->device];
			curtune = &curdeviceptr->dimensions;
			CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_dmu_numer.ptr), blocks[sblock].dev_dmu_numer.pitch/sizeof(real),
				(real*)(curblockptr->dev_dmu_numer.ptr), curblockptr->dev_dmu_numer.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_dmu_denom.ptr), blocks[sblock].dev_dmu_denom.pitch/sizeof(real),
				(real*)(curblockptr->dev_dmu_denom.ptr), curblockptr->dev_dmu_denom.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_dbeta_numer.ptr), blocks[sblock].dev_dbeta_numer.pitch/sizeof(real),
				(real*)(curblockptr->dev_dbeta_numer.ptr), curblockptr->dev_dbeta_numer.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_dbeta_denom.ptr), blocks[sblock].dev_dbeta_denom.pitch/sizeof(real),
				(real*)(curblockptr->dev_dbeta_denom.ptr), curblockptr->dev_dbeta_denom.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_drho_numer.ptr), blocks[sblock].dev_drho_numer.pitch/sizeof(real),
				(real*)(curblockptr->dev_drho_numer.ptr), curblockptr->dev_drho_numer.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_dlambda_numer.ptr), blocks[sblock].dev_dlambda_numer.pitch/sizeof(real),
				(real*)(curblockptr->dev_dlambda_numer.ptr), curblockptr->dev_dlambda_numer.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_dkappa_numer.ptr), blocks[sblock].dev_dkappa_numer.pitch/sizeof(real),
				(real*)(curblockptr->dev_dkappa_numer.ptr), curblockptr->dev_dkappa_numer.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_usum.ptr), blocks[sblock].dev_usum.pitch/sizeof(real),
				(real*)(curblockptr->dev_usum.ptr), curblockptr->dev_usum.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snmmodels_12dim.nblocks,
				curtune->c1c1snmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_dsigma2_numer.ptr), blocks[sblock].dev_dsigma2_numer.pitch/sizeof(real),
				(real*)(curblockptr->dev_dsigma2_numer.ptr), curblockptr->dev_dsigma2_numer.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snmmodels_12dim.nblocks,
				curtune->c1c1snmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_cnew.ptr), blocks[sblock].dev_cnew.pitch/sizeof(real),
				(real*)(curblockptr->dev_cnew.ptr), curblockptr->dev_cnew.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snchansnmmodels_12dim.nblocks,
				curtune->c1c1snchansnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(blocks[sblock].dev_phi.ptr), blocks[sblock].dev_phi.pitch/sizeof(real),
				(real*)(curblockptr->dev_phi.ptr), curblockptr->dev_phi.pitch/sizeof(real)
			);

		}


		/*
		 * Block 0 gets all the acumulated data.
		 */
		sblock = 0;
		model_block_t * sblockptr = &blocks[sblock];
		device_t * sdeviceptr = &gpus[sblockptr->device];
		curtune = &curdeviceptr->dimensions;

		cudaExtent nchannsdmnmmodels;
		nchannsdmnmmodels.width = currentConfig.nchannels * sizeof(real);
		nchannsdmnmmodels.height = currentConfig.nsdm;
		nchannsdmnmmodels.depth = currentConfig.nmmodels;

		cudaExtent nchannnchannnmmodels;
		nchannnchannnmmodels.width = currentConfig.nchannels * sizeof(real);
		nchannnchannnmmodels.height = currentConfig.nchannels;
		nchannnchannnmmodels.depth = currentConfig.nmmodels;


		CUDA_CHECK_RETURN(cudaSetDevice(sdeviceptr->realdevice));

		for (natural curblock = 1; curblock < devcount; curblock++) {
			curblockptr = &blocks[curblock];
			curdeviceptr = &gpus[curblockptr->device];
			cudaMemcpy3DPeerParms parms = {0};
			parms.extent = nchannsdmnmmodels;
			parms.dstDevice = sdeviceptr->realdevice;

			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_dmu_numer;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_dmu_numer.ptr), blocks[sblock].dev_dmu_numer.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);
			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_dmu_denom;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_dmu_denom.ptr), blocks[sblock].dev_dmu_denom.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);
			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_dbeta_numer;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_dbeta_numer.ptr), blocks[sblock].dev_dbeta_numer.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);
			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_dbeta_denom;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_dbeta_denom.ptr), blocks[sblock].dev_dbeta_denom.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);
			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_drho_numer;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_drho_numer.ptr), blocks[sblock].dev_drho_numer.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);
			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_dlambda_numer;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_dlambda_numer.ptr), blocks[sblock].dev_dlambda_numer.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);
			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_dkappa_numer;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_dkappa_numer.ptr), blocks[sblock].dev_dkappa_numer.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);

			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_usum;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_usum.ptr), blocks[sblock].dev_usum.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);

			parms.extent = nchannnchannnmmodels;
			parms.dstPtr = sdeviceptr->rnxnwork;
			parms.srcDevice = curdeviceptr->realdevice;
			parms.srcPtr = curblockptr->dev_phi;
			CUDA_CHECK_RETURN(cudaMemcpy3DPeerAsync(&parms,sdeviceptr->stream));
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snchansnmmodels_12dim.nblocks,
				curtune->c1c1snchansnmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_phi.ptr), blocks[sblock].dev_phi.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);


			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
					sdeviceptr->rnxnwork.ptr,
					sdeviceptr->rnxnwork.pitch,
					blocks[sblock].dev_dsigma2_numer.ptr,
					blocks[sblock].dev_dsigma2_numer.pitch,
					blocks[sblock].dev_dsigma2_numer.xsize * sizeof(real),
					blocks[sblock].dev_dsigma2_numer.ysize,
					cudaMemcpyDeviceToDevice,
					curdeviceptr->stream
					));

			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snmmodels_12dim.nblocks,
				curtune->c1c1snmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_dsigma2_numer.ptr), blocks[sblock].dev_dsigma2_numer.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);

			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
					sdeviceptr->rnxnwork.ptr,
					sdeviceptr->rnxnwork.pitch,
					blocks[sblock].dev_cnew.ptr,
					blocks[sblock].dev_cnew.pitch,
					blocks[sblock].dev_cnew.xsize * sizeof(real),
					blocks[sblock].dev_cnew.ysize,
					cudaMemcpyDeviceToDevice,
					curdeviceptr->stream
					));

			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snmmodels_12dim.nblocks,
				curtune->c1c1snmmodels_12dim.nthreads,
				0,
				sdeviceptr->stream,
				(real*)(blocks[sblock].dev_cnew.ptr), blocks[sblock].dev_cnew.pitch/sizeof(real),
				(real*)(sdeviceptr->rnxnwork.ptr), sdeviceptr->rnxnwork.pitch/sizeof(real)
			);

		}

		DEVICE_SYNC
		DBUILD(3,
		{
		natural curblock = 0;
		curblockptr = &blocks[curblock];
		curdeviceptr = &gpus[curblockptr->device];
		CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dmu_numer, nmmodels, "datasets/mu_numer_all.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dmu_denom, nmmodels, "datasets/mu_denom_all.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dbeta_numer, nmmodels, "datasets/beta_numer_all.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dbeta_denom, nmmodels, "datasets/beta_denom_all.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_drho_numer, nmmodels, "datasets/rho_numer_all.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dlambda_numer, nmmodels, "datasets/lambda_numer_all.fdt")
		DDEVWRITEMAT(curdeviceptr->stream, curblockptr->dev_dsigma2_numer,  "datasets/sigma2_numer_all.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_dkappa_numer, nmmodels, "datasets/kappa_numer_all.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_phi, nmmodels, "datasets/phi_all.fdt")
		DDEVWRITEMAT(curdeviceptr->stream, curblockptr->dev_cnew, "datasets/cnew_all.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_usum, nmmodels, "datasets/usum_all.fdt")
		})

//		{	// get usumsum
//		natural curblock = 0;
//		curblockptr = &blocks[curblock];
//		curdeviceptr = &gpus[curblockptr->device];
//		CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
//		cudaMemcpy3DParms parms = {0};
//		parms.extent = nchannsdmnmmodels;
//		parms.dstPtr = curblockptr->host_usum;
//		parms.srcPtr = curblockptr->dev_usum;
//		parms.kind = cudaMemcpyDeviceToHost;
//		CUDA_CHECK_RETURN(cudaMemcpy3D(&parms));
//		for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
//			curmodelptr = &mmodels[curmodel];
//			curmodelptr->host_usumsum = 0.0;
//			for (natural cursdm = 0; cursdm < nsdm; cursdm++) {
//				curmodelptr->host_usumsum += cblas_dasum(nchannels, OFFSET3D1D(curblockptr->host_usum, cursdm, curmodel), 1);
//			}
//			fprintf(stdout, "Model %d => host_usumsum %.16f\n", curmodel, curmodelptr->host_usumsum);
//		}
//		}

		for (natural curblock = 0; curblock < nblocks; curblock++) {
			curblockptr = &blocks[curblock];
			for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
				curmodelptr = &mmodels[curmodel];
				curmodelptr->host_vsumsum += ((real*)(curblockptr->host_vsum.ptr))[curmodel];
//				fprintf(stdout, "block %d mmodel %d vsum %.16f\n", curblock, curmodel, ((real*)(curblockptr->host_vsum.ptr))[curmodel]);
			}
		}
		for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
			fprintf(stdout, "mmodel %lu -> vsum %.16f\n", curmodel, mmodels[curmodel].host_vsumsum);
			DBUILD(3,
			{char nombre[80];
			sprintf(nombre, "datasets/vsumvalue-m%d.fdt", curmodel);
			C_CHECK_RETURN(writeValue(mmodels[curmodel].host_vsumsum, nombre));
			}
			)
		}

		host_ll[iter] = host_ll[iter] / (nchannels*nsamples);

		fprintf(stdout, "Iteration %lu: lrate %.16f LL %.16f \n", iter, lrate, host_ll[iter]);

		if (iter > 1 && host_ll[iter] < host_ll[iter-1]) {
			fprintf(stdout, "Likelihood decreasing!\n");
			lrate = lrate * lratefact;
			numdec = numdec + 1;
			if (numdec > maxdec) {
				lratemax = lratemax * lratefact;
				numdec = 0;
			}
		}

		if (currentConfig.update_gm) {
			for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
				curmodelptr = &mmodels[curmodel];
				curmodelptr->gm = curmodelptr->host_vsumsum/nsamples;
			}
		}

		natural curblock = 0;
		curblockptr = &blocks[curblock];
		curdeviceptr = &gpus[curblockptr->device];
		CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
		if (currentConfig.update_alpha) {
			for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
				curmodelptr = &mmodels[curmodel];
				alpha = 1.0/curmodelptr->host_vsumsum;
				beta = 0.0;
				CUBLAS_CHECK_RETURN(
					cublasDgeam(
					curdeviceptr->cublasHandle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					curmodelptr->dev_alpha.xsize, curmodelptr->dev_alpha.ysize,
					&alpha,
					OFFSET3D(curblockptr->dev_usum, curmodel), curblockptr->dev_usum.pitch/sizeof(real),
					&beta, OFFSET3D(curblockptr->dev_usum, curmodel), curblockptr->dev_usum.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->alpha, curmodel), curdeviceptr->alpha.pitch/sizeof(real)
				));
			}
			DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->alpha, nmmodels, "datasets/new_alpha.fdt")
		}

		for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
			curmodelptr = &mmodels[curmodel];
			alpha = 1.0/curmodelptr->host_vsumsum;
			CUBLAS_CHECK_RETURN(
				cublasDscal_v2(
				curdeviceptr->cublasHandle,
				curblockptr->dev_dsigma2_numer.xsize,
				&alpha,
				OFFSET2D(curblockptr->dev_dsigma2_numer, curmodel), 1
			));
			CUBLAS_CHECK_RETURN(
				cublasDscal_v2(
				curdeviceptr->cublasHandle,
				curblockptr->dev_cnew.xsize,
				&alpha,
				OFFSET2D(curblockptr->dev_cnew, curmodel), 1
			));
			beta = 0.0;
			CUBLAS_CHECK_RETURN(
				cublasDgeam(
				curdeviceptr->cublasHandle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				curblockptr->dev_phi.xsize, curblockptr->dev_phi.ysize,
				&alpha,
				OFFSET3D(curblockptr->dev_phi, curmodel), curblockptr->dev_phi.pitch/sizeof(real),
				&beta, OFFSET3D(curblockptr->dev_phi, curmodel), curblockptr->dev_phi.pitch/sizeof(real),
				OFFSET3D(curblockptr->dev_phi, curmodel), curblockptr->dev_phi.pitch/sizeof(real)
			));
		}

		CUDA_CHECK_RETURN(cudaMemset2DAsync(
			curdeviceptr->rnxbswork.ptr,
			curdeviceptr->rnxbswork.pitch,
			0,
			curdeviceptr->rnxbswork.xsize * sizeof(real),
			nmmodels,
			curdeviceptr->stream
		));
		CUDA_CHECK_RETURN(cudaMemset2DAsync(
			curdeviceptr->rnxbswork2.ptr,
			curdeviceptr->rnxbswork2.pitch,
			0,
			curdeviceptr->rnxbswork2.xsize * sizeof(real),
			nmmodels,
			curdeviceptr->stream
		));

		for (natural cursdm = 0; cursdm < nsdm; cursdm ++) {
			CUDA_KERNEL_CALL(
				updatekappalambda,
				curtune->c1c1snmmodels_11dim.nblocks,
				curtune->c1c1snmmodels_11dim.nthreads,
				0,
				curdeviceptr->stream,
				OFFSET2D(curblockptr->dev_dkappa_numer, cursdm), curblockptr->dev_dkappa_numer.pitch/sizeof(real) * nsdm,
				OFFSET2D(curblockptr->dev_usum, cursdm), curblockptr->dev_usum.pitch/sizeof(real) * nsdm,
				OFFSET2D(curblockptr->dev_dlambda_numer, cursdm), curblockptr->dev_dlambda_numer.pitch/sizeof(real) * nsdm,
				(real*)curdeviceptr->rnxbswork.ptr, curdeviceptr->rnxbswork.pitch/sizeof(real),
				(real*)curdeviceptr->rnxbswork2.ptr, curdeviceptr->rnxbswork2.pitch/sizeof(real),
				OFFSET2D(curdeviceptr->alpha, cursdm), curdeviceptr->alpha.pitch/sizeof(real) * nsdm,
				OFFSET2D(curdeviceptr->mu, cursdm), curdeviceptr->mu.pitch/sizeof(real) * nsdm
			);
		}

		DDEVWRITEMAT(curdeviceptr->stream, curblockptr->dev_dsigma2_numer, "datasets/new_sigma2.fdt")
		DDEVWRITEMAT(curdeviceptr->stream, curblockptr->dev_cnew, "datasets/new_cnew.fdt")
		DDEVWRITEMAT3D(curdeviceptr->stream, curblockptr->dev_phi, nmmodels, "datasets/new_phi.fdt")
		DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->rnxbswork, "datasets/kappa.fdt")
		DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->rnxbswork2, "datasets/lambda.fdt")

		if (currentConfig.update_mu) {
			CUDA_KERNEL_CALL(
				divide,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(curblockptr->dev_dmu_numer.ptr), curblockptr->dev_dmu_numer.pitch/sizeof(real),
				(real*)(curblockptr->dev_dmu_denom.ptr), curblockptr->dev_dmu_denom.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				acumulate,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(curdeviceptr->mu.ptr), curdeviceptr->mu.pitch/sizeof(real),
				(real*)(curblockptr->dev_dmu_numer.ptr), curblockptr->dev_dmu_numer.pitch/sizeof(real)
			);
			DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->mu, nmmodels, "datasets/new_mu.fdt")
		}

		if (currentConfig.update_beta) {
			CUDA_KERNEL_CALL(
				divide,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(curblockptr->dev_dbeta_numer.ptr), curblockptr->dev_dbeta_numer.pitch/sizeof(real),
				(real*)(curblockptr->dev_dbeta_denom.ptr), curblockptr->dev_dbeta_denom.pitch/sizeof(real)
			);
			CUDA_KERNEL_CALL(
				multiply,
				curtune->c1c1snsdmnmmodels_12dim.nblocks,
				curtune->c1c1snsdmnmmodels_12dim.nthreads,
				0,
				curdeviceptr->stream,
				(real*)(curdeviceptr->beta.ptr), curdeviceptr->beta.pitch/sizeof(real),
				(real*)(curblockptr->dev_dbeta_numer.ptr), curblockptr->dev_dbeta_numer.pitch/sizeof(real)
			);
			DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->beta, nmmodels, "datasets/new_beta.fdt")
		}

		if (currentConfig.update_rho && iter >= currentConfig.rho_start_iter) {
			for (natural cursdm = 0; cursdm < nsdm; cursdm ++) {
				updaterho<<<
					curtune->c1c1snmmodels_11dim.nblocks,
					curtune->c1c1snmmodels_11dim.nthreads,
					0,
					curdeviceptr->stream>>>(
					OFFSET2D(curdeviceptr->rho, cursdm), curdeviceptr->rho.pitch/sizeof(real)  * nsdm,
					OFFSET2D(curblockptr->dev_drho_numer, cursdm), curblockptr->dev_drho_numer.pitch/sizeof(real)  * nsdm,
					OFFSET2D(curblockptr->dev_usum, cursdm), curblockptr->dev_usum.pitch/sizeof(real) * nsdm,
					currentConfig.rholrate, currentConfig.rhomin, currentConfig.rhomax
				);
				CUDA_CHECK_LAST();
			}
			DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->rho, nmmodels, "datasets/new_rho.fdt")
		}
		unsigned int * bflag = new unsigned int[nmmodels];
		if (currentConfig.update_A) {
			int no_newt = 0;
			for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
				curmodelptr = &mmodels[curmodel];
				/*
				 * rnwork = kappa .* sigma2
				 */
				CUDA_KERNEL_CALL(
					multiplyTo,
					curtune->vchannels_11dim.nblocks,
					curtune->vchannels_11dim.nthreads,
					0,
					curdeviceptr->stream,
					OFFSET2D(curdeviceptr->rnxbswork, curmodel), curdeviceptr->rnxbswork.pitch/sizeof(real),
					OFFSET2D(curblockptr->dev_dsigma2_numer, curmodel), curblockptr->dev_dsigma2_numer.pitch/sizeof(real),
					(real*)(curdeviceptr->rnwork.ptr), curdeviceptr->rnwork.pitch/sizeof(real)
				);
				/*
				 * denoms = kappa .* sigma2 .* kappa' .* sigma2'
				 */
				alpha = 1.0;
				beta = 0.0;
				CUBLAS_CHECK_RETURN(cublasDgemm_v2(
					curdeviceptr->cublasHandle,
					CUBLAS_OP_T,
					CUBLAS_OP_N,
					curdeviceptr->rnwork.xsize,
					curdeviceptr->rnwork.xsize,
					1,
					&alpha,
					(real*)curdeviceptr->rnwork.ptr, 1,
					(real*)curdeviceptr->rnwork.ptr, 1,
					&beta,
					(real*)curdeviceptr->denoms.ptr, curdeviceptr->denoms.pitch/sizeof(real)

				));

				bflag[curmodel] = 0;
				CUDA_CHECK_RETURN(cudaMemcpyAsync(curdeviceptr->rbswork.ptr, &bflag[curmodel] , sizeof(bflag[curmodel]), cudaMemcpyHostToDevice, curdeviceptr->stream));
				CUDA_KERNEL_CALL(
					getbflag,
					curtune->c1c1c_11dim.nblocks,
					curtune->c1c1c_11dim.nthreads,
					0,
					curdeviceptr->stream,
					(real*)curdeviceptr->denoms.ptr, curdeviceptr->denoms.pitch/sizeof(real),
					(unsigned int*)curdeviceptr->rbswork.ptr
				);
				DBUILD(3,
				{char nombre[80];
				sprintf(nombre, "datasets/denoms-m%lu.fdt", curmodel);
				DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->denoms, nombre)
				})


				CUDA_CHECK_RETURN(cudaMemcpy(&bflag[curmodel],curdeviceptr->rbswork.ptr, sizeof(bflag[curmodel]), cudaMemcpyDeviceToHost));
				DPRINTF(1, "Model %lu -> BFLAG %d\n", curmodel, bflag[curmodel]);

				if (bflag[curmodel]  > 0) {
					no_newt = 1;
				}

				if (bflag[curmodel]  == 0 && currentConfig.do_newton == 1 && iter >= currentConfig.newt_start_iter) {
					/*
					 * noms = kappa * sigma2
					 */
					alpha = -1.0;
					beta = 0.0;
					CUBLAS_CHECK_RETURN(cublasDgemm_v2(
						curdeviceptr->cublasHandle,
						CUBLAS_OP_T,
						CUBLAS_OP_N,
						curdeviceptr->rnxbswork.xsize,
						curblockptr->dev_dsigma2_numer.xsize,
						1,
						&alpha,
						OFFSET2D(curblockptr->dev_dsigma2_numer, curmodel), 1,
						OFFSET2D(curdeviceptr->rnxbswork, curmodel), 1,
						&beta,
						(real*)curdeviceptr->noms.ptr, curdeviceptr->noms.pitch/sizeof(real)
					));

					CUDA_KERNEL_CALL(
						getB,
						curtune->c1c1c_11dim.nblocks,
						curtune->c1c1c_11dim.nthreads,
						0,
						curdeviceptr->stream,
						OFFSET3D(curblockptr->dev_phi, curmodel), curblockptr->dev_phi.pitch/sizeof(real),
						(real*)curdeviceptr->noms.ptr, curdeviceptr->noms.pitch/sizeof(real),
						(real*)curdeviceptr->denoms.ptr, curdeviceptr->denoms.pitch/sizeof(real),
						OFFSET3D(curdeviceptr->B, curmodel), curdeviceptr->B.pitch/sizeof(real)
					);
					CUDA_KERNEL_CALL(
						updateBDiagonal,
						curtune->c1c1c_11dim.nblocks,
						curtune->c1c1c_11dim.nthreads,
						0,
						curdeviceptr->stream,
						OFFSET3D(curblockptr->dev_phi, curmodel), curblockptr->dev_phi.pitch/sizeof(real),
						OFFSET2D(curdeviceptr->rnxbswork2, curmodel),
						OFFSET3D(curdeviceptr->B, curmodel), curdeviceptr->B.pitch/sizeof(real)
					);
					DBUILD(3,
					{char nombre[80];
						sprintf(nombre, "datasets/noms-m%lu.fdt", curmodel);
						DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->noms, nombre)
						if (curmodel == (nmmodels-1)) {
							DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->B, nmmodels, "datasets/B.fdt")
						}
					})

				} else {
					CUDA_KERNEL_CALL(
						geteyemphi,
						curtune->c1c1c_11dim.nblocks,
						curtune->c1c1c_11dim.nthreads,
						0,
						curdeviceptr->stream,
						OFFSET3D(curblockptr->dev_phi, curmodel), curblockptr->dev_phi.pitch/sizeof(real),
						OFFSET3D(curdeviceptr->B, curmodel), curdeviceptr->B.pitch/sizeof(real)
					);
				}
			}

			if (iter >= currentConfig.newt_start_iter &&  currentConfig.do_newton && (no_newt == 0)) {
				lrate = min(currentConfig.lratemax, lrate + min(0.1, lrate));
			} else {
				lrate = min(currentConfig.lnatrate, lrate + min(0.1, lrate));
			}
			DPRINTF(1, "LRATE %.16f\n", lrate);

			for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
				curmodelptr = &mmodels[curmodel];
				CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
					curdeviceptr->atmp.ptr,
					curdeviceptr->atmp.pitch,
					curmodelptr->dev_a.ptr,
					curmodelptr->dev_a.pitch,
					curmodelptr->dev_a.xsize * sizeof(real),
					curmodelptr->dev_a.ysize,
					cudaMemcpyDeviceToDevice,
					curdeviceptr->stream
				));

				CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
					OFFSET3D(curdeviceptr->acopy, curmodel),
					curdeviceptr->acopy.pitch,
					curdeviceptr->atmp.ptr,
					curdeviceptr->atmp.pitch,
					nchannels * sizeof(real),
					nchannels,
					cudaMemcpyDeviceToDevice,
					curdeviceptr->stream
				));
				DBUILD(3,
				{char nombre[80];
				sprintf(nombre, "datasets/A-m%lu.fdt", curmodel);
				DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->atmp, nombre)
				})

				if (bflag[curmodel] == 0 && currentConfig.do_newton == 1 && iter >= currentConfig.newt_start_iter) {
					// A = A + lrate * A * B
					alpha = lrate;
					beta = 1.0;
				} else {
					if (bflag[curmodel] != 0 && iter >= currentConfig.newt_start_iter) {
						fprintf(stdout, "Hessian non positive definite! Using natural gradient!\n");
					}
					alpha = -lrate;
					beta = 1.0;
					// A = A - lrate * A * B
				}

				CUBLAS_CHECK_RETURN(cublasDgemm_v2(
					curdeviceptr->cublasHandle,
					CUBLAS_OP_N,
					CUBLAS_OP_N,
					curmodelptr->dev_a.ysize,
					curmodelptr->dev_a.xsize,
					curdeviceptr->B.ysize,
					&alpha,
					(real*) curdeviceptr->atmp.ptr, curdeviceptr->atmp.pitch/sizeof(real),
					OFFSET3D(curdeviceptr->B, curmodel), curdeviceptr->B.pitch/sizeof(real),
					&beta,
					OFFSET3D(curdeviceptr->acopy, curmodel), curdeviceptr->acopy.pitch/sizeof(real)
				));
				DBUILD(3,
				{
					CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
						curdeviceptr->atmp.ptr,
						curdeviceptr->atmp.pitch,
						OFFSET3D(curdeviceptr->acopy, curmodel),
						curdeviceptr->acopy.pitch,
						nchannels * sizeof(real),
						nchannels,
						cudaMemcpyDeviceToDevice,
						curdeviceptr->stream
					));
					char nombre[80];
					if (bflag[curmodel] == 0) {
						sprintf(nombre, "datasets/A-b0-m%lu.fdt", curmodel);
					} else {
						sprintf(nombre, "datasets/A-b1-m%lu.fdt", curmodel);
					}
					DDEVWRITEMAT(curdeviceptr->stream, curdeviceptr->atmp, nombre)
				})

			}


			if (currentConfig.do_reparm) {
				for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
					curmodelptr = &mmodels[curmodel];
					CUDA_KERNEL_CALL(normalizeAsave,
						curdeviceptr->dimensions.c1c1c_11dim.nblocks, curdeviceptr->dimensions.c1c1c_11dim.nthreads,
						nchannels * sizeof(real), curdeviceptr->stream,
						OFFSET3D(curdeviceptr->acopy, curmodel), curdeviceptr->acopy.pitch/sizeof(real),
						(real*) curdeviceptr->rbswork.ptr
					);

					CUDA_KERNEL_CALL(normalizemubeta,
						curdeviceptr->dimensions.c1c1snsdm_11dim.nblocks, curdeviceptr->dimensions.c1c1snsdm_11dim.nthreads,
						0, curdeviceptr->stream,
						(real*) curdeviceptr->rbswork.ptr,
						OFFSET3D(curdeviceptr->mu, curmodel), curdeviceptr->mu.pitch/sizeof(real),
						OFFSET3D(curdeviceptr->beta, curmodel), curdeviceptr->beta.pitch/sizeof(real)
					);
					CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
						OFFSET3D(curdeviceptr->W, curmodel),
						curdeviceptr->W.pitch,
						curmodelptr->host_pinvv.ptr,
						curmodelptr->host_pinvv.pitch,
						curmodelptr->host_pinvv.xsize * sizeof(real),
						curmodelptr->host_pinvv.ysize,
						cudaMemcpyHostToDevice,
						curdeviceptr->stream
					));
				}
				DBUILD(3,
				{
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->acopy, nmmodels, "datasets/Anorm.fdt")
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->mu, nmmodels, "datasets/munorm.fdt")
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->beta, nmmodels, "datasets/betanorm.fdt")
				})

				CUDA_KERNEL_CALL(
					cnewminusc,
					curtune->c1c1snmmodels_11dim.nblocks,
					curtune->c1c1snmmodels_11dim.nthreads,
					0, curdeviceptr->stream,
					(real*) curdeviceptr->c.ptr, curdeviceptr->c.pitch/sizeof(real),
					(real*) curblockptr->dev_cnew.ptr, curblockptr->dev_cnew.pitch/sizeof(real)
				);
				for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
					alpha = 1.0;
					beta = 0.0;
					/* W * (cnew - c) */
					CUBLAS_CHECK_RETURN(cublasDgemv_v2(
						curdeviceptr->cublasHandle,
						CUBLAS_OP_N,
						curdeviceptr->W.ysize,
						curdeviceptr->W.xsize,
						&alpha,
						OFFSET3D(curdeviceptr->W,curmodel),
						curdeviceptr->W.pitch/sizeof(real),
						OFFSET2D(curdeviceptr->c,curmodel),
						1,
						&beta,
						(real*)curdeviceptr->rnwork.ptr,
						1
					));
					CUDA_KERNEL_CALL(
						substract,
						curtune->c1c1snsdm_11dim.nblocks,
						curtune->c1c1snsdm_11dim.nthreads,
						nchannels * sizeof(real),
						curdeviceptr->stream,
						OFFSET3D(curdeviceptr->mu,curmodel),
						curdeviceptr->mu.pitch/sizeof(real),
						(real*)curdeviceptr->rnwork.ptr
					);


				}
				DBUILD(3,
				{
					DDEVWRITEMAT3D(curdeviceptr->stream, curdeviceptr->mu, nmmodels, "datasets/lastmu.fdt")
				})


			}
		}
		delete bflag;

		for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
			curmodelptr =  &mmodels[curmodel];
			/*
			 * Copy c, beta, alpha, mu and rho to all the devices.
			 */
			CUDA_CHECK_RETURN(cudaMemcpyAsync(
				curmodelptr->dev_c.ptr,
				OFFSET2D(curblockptr->dev_cnew, curmodel),
				curmodelptr->dev_c.xsize * sizeof(real),
				cudaMemcpyDeviceToDevice,
				curdeviceptr->stream
			));
			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
				curmodelptr->dev_beta.ptr,
				curmodelptr->dev_beta.pitch,
				OFFSET3D(curdeviceptr->beta, curmodel),
				curdeviceptr->beta.pitch,
				curmodelptr->dev_beta.xsize * sizeof(real),
				curmodelptr->dev_beta.ysize,
				cudaMemcpyDeviceToDevice,
				curdeviceptr->stream
			));
			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
				curmodelptr->dev_mu.ptr,
				curmodelptr->dev_mu.pitch,
				OFFSET3D(curdeviceptr->mu, curmodel),
				curdeviceptr->mu.pitch,
				curmodelptr->dev_mu.xsize * sizeof(real),
				curmodelptr->dev_mu.ysize,
				cudaMemcpyDeviceToDevice,
				curdeviceptr->stream
			));
			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
				curmodelptr->dev_alpha.ptr,
				curmodelptr->dev_alpha.pitch,
				OFFSET3D(curdeviceptr->alpha, curmodel),
				curdeviceptr->alpha.pitch,
				curmodelptr->dev_alpha.xsize * sizeof(real),
				curmodelptr->dev_alpha.ysize,
				cudaMemcpyDeviceToDevice,
				curdeviceptr->stream
			));
			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
				curmodelptr->dev_rho.ptr,
				curmodelptr->dev_rho.pitch,
				OFFSET3D(curdeviceptr->rho, curmodel),
				curdeviceptr->rho.pitch,
				curmodelptr->dev_rho.xsize * sizeof(real),
				curmodelptr->dev_rho.ysize,
				cudaMemcpyDeviceToDevice,
				curdeviceptr->stream
			));
			CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
				curmodelptr->dev_a.ptr,
				curmodelptr->dev_a.pitch,
				OFFSET3D(curdeviceptr->acopy, curmodel),
				curdeviceptr->acopy.pitch,
				curmodelptr->dev_a.xsize * sizeof(real),
				curmodelptr->dev_a.ysize,
				cudaMemcpyDeviceToDevice,
				curdeviceptr->stream
			));

#ifdef ITERTEST
			{char nombre[80];
				CUDA_CHECK_RETURN(cudaSetDevice(gpus[curmodelptr->master_device].realdevice));
				CUDA_CHECK_RETURN(cudaStreamSynchronize(gpus[curmodelptr->master_device].stream));
				sprintf(nombre, "datasets/FINAL-A-i%lu-m%lu.fdt", iter, curmodel);
				C_CHECK_RETURN(writeDevMatrix(curmodelptr->dev_a, nombre));
				sprintf(nombre, "datasets/FINAL-RHO-i%lu-m%lu.fdt", iter, curmodel);
				C_CHECK_RETURN(writeDevMatrix(curmodelptr->dev_rho, nombre));
				sprintf(nombre, "datasets/FINAL-MU-i%lu-m%lu.fdt", iter, curmodel);
				C_CHECK_RETURN(writeDevMatrix(curmodelptr->dev_mu, nombre));
				sprintf(nombre, "datasets/FINAL-BETA-i%lu-m%lu.fdt", iter, curmodel);
				C_CHECK_RETURN(writeDevMatrix(curmodelptr->dev_beta, nombre));
				sprintf(nombre, "datasets/FINAL-ALPHA-i%lu-m%lu.fdt", iter, curmodel);
				C_CHECK_RETURN(writeDevMatrix(curmodelptr->dev_alpha, nombre));
				sprintf(nombre, "datasets/FINAL-C-i%lu-m%lu.fdt", iter, curmodel);
				C_CHECK_RETURN(writeDevMatrix(curmodelptr->dev_c,  nombre));
				CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
			}
#endif



		}

		SECTIMER_END(iteration);
		TICKTIMER_END(iteration);
		fprintf(stdout, "Iteration %lu times = %lu ticks, %lu ms \n", iter, iteration_ticks, iteration_msecs);
	}
	curdeviceptr = &gpus[0];
	CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));

	/* Save W */
	cudaPitchedPtr resultW;
	resultW.xsize = nchannels;
	resultW.pitch = nchannels * sizeof(real);
	resultW.ysize = nchannels * nmmodels;
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultW.ptr, resultW.pitch * resultW.ysize, cudaHostAllocDefault));

	for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
			curmodelptr =  &mmodels[curmodel];
			CUDA_CHECK_RETURN(cudaMemcpy2D(
				HOFFSET3D(resultW.ptr, resultW.pitch, nchannels, curmodel),
				resultW.pitch,
				curmodelptr->host_pinvv.ptr,
				curmodelptr->host_pinvv.pitch,
				curmodelptr->host_pinvv.xsize * sizeof(real),
				curmodelptr->host_pinvv.ysize,
				cudaMemcpyHostToHost));
	}

	model_t * model0 = &mmodels[0];
	CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
		model0->host_pinva.ptr,
		model0->host_pinva.pitch,
		curdeviceptr->sphere.ptr,
		curdeviceptr->sphere.pitch,
		curdeviceptr->sphere.xsize * sizeof(real),
		curdeviceptr->sphere.ysize,
		cudaMemcpyDeviceToHost,
		curdeviceptr->stream
	));
	pinv(model0);


	for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
		curmodelptr =  &mmodels[curmodel];
		CUDA_CHECK_RETURN(cudaSetDevice(curmodelptr->master_device));
		/* c = pinvv * c + xmn */
		CUDA_CHECK_RETURN(cudaMemcpyAsync(
			curdeviceptr->rnwork.ptr,
			curdeviceptr->means.ptr,
			curdeviceptr->means.xsize * sizeof(real),
			cudaMemcpyDeviceToDevice,
			curdeviceptr->stream
		));

		CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
			curdeviceptr->rnxbswork.ptr,
			curdeviceptr->rnxbswork.pitch,
			model0->host_pinvv.ptr,
			model0->host_pinvv.pitch,
			model0->host_pinvv.xsize * sizeof(real),
			model0->host_pinvv.ysize,
			cudaMemcpyHostToDevice,
			curdeviceptr->stream
		));

		alpha = 1.0;
		beta = 1.0;
		CUBLAS_CHECK_RETURN(cublasDgemv_v2(
			curdeviceptr->cublasHandle,
			CUBLAS_OP_N,
			curdeviceptr->rnxbswork.xsize,
			curdeviceptr->rnxbswork.xsize,
			&alpha,
			(real *) curdeviceptr->rnxbswork.ptr,
			curdeviceptr->rnxbswork.pitch/sizeof(real),
			(real *) curmodelptr->dev_c.ptr, 1,
			&beta,
			(real *) curdeviceptr->rnwork.ptr,
			1
		));
		CUDA_CHECK_RETURN(cudaMemcpyAsync(
			curmodelptr->dev_c.ptr,
			curdeviceptr->rnwork.ptr,
			curdeviceptr->c.xsize * sizeof(real),
			cudaMemcpyDeviceToDevice,
			curdeviceptr->stream
		));

		/* A = pinvv * a */

		/* Copy A into atmp */
		CUDA_CHECK_RETURN(cudaMemcpy2DAsync(
			curdeviceptr->atmp.ptr,
			curdeviceptr->atmp.pitch,
			curmodelptr->dev_a.ptr,
			curmodelptr->dev_a.pitch,
			nchannels * sizeof(real),
			nchannels,
			cudaMemcpyDeviceToDevice,
			curdeviceptr->stream
		));
		alpha = 1.0;
		beta = 0.0;
		CUBLAS_CHECK_RETURN(cublasDgemm_v2(
			curdeviceptr->cublasHandle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			nchannels,
			nchannels,
			nchannels,
			&alpha,
			(real*) curdeviceptr->rnxbswork.ptr, curdeviceptr->rnxbswork.pitch/sizeof(real),
			(real*) curdeviceptr->atmp.ptr, curdeviceptr->atmp.pitch/sizeof(real),
			&beta,
			(real*) curmodelptr->dev_a.ptr, curmodelptr->dev_a.pitch/sizeof(real)
		));


		CUDA_KERNEL_CALL(normalizeAsave,
			curdeviceptr->dimensions.c1c1c_11dim.nblocks, curdeviceptr->dimensions.c1c1c_11dim.nthreads,
			nchannels * sizeof(real), curdeviceptr->stream,
			(real*) curmodelptr->dev_a.ptr, curmodelptr->dev_a.pitch/sizeof(real),
			(real*) curdeviceptr->rbswork.ptr
		);

		CUDA_KERNEL_CALL(normalizemubeta,
			curdeviceptr->dimensions.c1c1snsdm_11dim.nblocks, curdeviceptr->dimensions.c1c1snsdm_11dim.nthreads,
			0, curdeviceptr->stream,
			(real*) curdeviceptr->rbswork.ptr,
			OFFSET3D(curdeviceptr->mu, curmodel), curdeviceptr->mu.pitch/sizeof(real),
			OFFSET3D(curdeviceptr->beta, curmodel), curdeviceptr->beta.pitch/sizeof(real)
		);
	}

	fprintf(stdout, "Finished!\n");

	cudaPitchedPtr resultA;
	resultA.xsize = nchannels;
	resultA.pitch = nchannels * sizeof(real);
	resultA.ysize = nchannels * nmmodels;

	cudaPitchedPtr resultc;
	resultc.xsize = nchannels;
	resultc.pitch = nchannels * sizeof(real);
	resultc.ysize = nmmodels;

	cudaPitchedPtr resultLtall;
	resultLtall.xsize = nsamples;
	resultLtall.pitch = nsamples * sizeof(real);
	resultLtall.ysize = nmmodels;

	cudaPitchedPtr resultgm;
	resultgm.xsize = nmmodels;
	resultgm.pitch = nmmodels * sizeof(real);
	resultgm.ysize = 1;

	cudaPitchedPtr resultalpha;
	resultalpha.xsize = nchannels;
	resultalpha.pitch = nchannels * sizeof(real);
	resultalpha.ysize = nsdm * nmmodels;

	cudaPitchedPtr resultmu;
	resultmu.xsize = nchannels;
	resultmu.pitch = nchannels * sizeof(real);
	resultmu.ysize = nsdm * nmmodels;

	cudaPitchedPtr resultbeta;
	resultbeta.xsize = nchannels;
	resultbeta.pitch = nchannels * sizeof(real);
	resultbeta.ysize = nsdm * nmmodels;

	cudaPitchedPtr resultrho;
	resultrho.xsize = nchannels;
	resultrho.pitch = nchannels * sizeof(real);
	resultrho.ysize = nsdm * nmmodels;


	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultA.ptr, resultA.pitch * resultA.ysize, cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultc, resultc.pitch * resultc.ysize, cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultLtall, resultLtall.pitch * resultLtall.ysize, cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultgm, resultgm.pitch * resultgm.ysize, cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultalpha, resultalpha.pitch * resultalpha.ysize, cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultmu, resultmu.pitch * resultmu.ysize, cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultbeta, resultbeta.pitch * resultbeta.ysize, cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&resultrho, resultrho.pitch * resultrho.ysize, cudaHostAllocDefault));


	for (natural curmodel = 0; curmodel < nmmodels; curmodel++) {
		curmodelptr = &mmodels[curmodel];
		curdeviceptr = &gpus[curmodelptr->master_device];
		CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
		CUDA_CHECK_RETURN(cudaMemcpy2D(
			HOFFSET3D(resultA.ptr, resultA.pitch, nchannels, curmodel),
			resultA.pitch,
			curmodelptr->dev_a.ptr,
			curmodelptr->dev_a.pitch,
			curmodelptr->dev_a.xsize * sizeof(real),
			curmodelptr->dev_a.ysize,
			cudaMemcpyDeviceToHost));

		CUDA_CHECK_RETURN(cudaMemcpy2D(
			HOFFSET2D(resultc.ptr, resultc.pitch, curmodel),
			resultc.pitch,
			curmodelptr->dev_c.ptr,
			curmodelptr->dev_c.pitch,
			curmodelptr->dev_c.xsize * sizeof(real),
			curmodelptr->dev_c.ysize,
			cudaMemcpyDeviceToHost));


		CUDA_CHECK_RETURN(cudaMemcpy(
			HOFFSET2D(resultLtall.ptr, resultLtall.pitch, curmodel),
			curmodelptr->dev_ltall.ptr,
			nsamples * sizeof(real),
			cudaMemcpyDeviceToHost));

		((real*)resultgm.ptr)[curmodel] = curmodelptr->gm;

		CUDA_CHECK_RETURN(cudaMemcpy2D(
			HOFFSET3D(resultalpha.ptr, resultalpha.pitch, nsdm, curmodel),
			resultalpha.pitch,
			curmodelptr->dev_alpha.ptr,
			curmodelptr->dev_alpha.pitch,
			curmodelptr->dev_alpha.xsize * sizeof(real),
			curmodelptr->dev_alpha.ysize,
			cudaMemcpyDeviceToHost));

		CUDA_CHECK_RETURN(cudaMemcpy2D(
			HOFFSET3D(resultmu.ptr, resultmu.pitch, nsdm, curmodel),
			resultmu.pitch,
			curmodelptr->dev_mu.ptr,
			curmodelptr->dev_mu.pitch,
			curmodelptr->dev_mu.xsize * sizeof(real),
			curmodelptr->dev_mu.ysize,
			cudaMemcpyDeviceToHost));

		CUDA_CHECK_RETURN(cudaMemcpy2D(
			HOFFSET3D(resultbeta.ptr, resultbeta.pitch, nsdm, curmodel),
			resultbeta.pitch,
			curmodelptr->dev_beta.ptr,
			curmodelptr->dev_beta.pitch,
			curmodelptr->dev_beta.xsize * sizeof(real),
			curmodelptr->dev_beta.ysize,
			cudaMemcpyDeviceToHost));

		CUDA_CHECK_RETURN(cudaMemcpy2D(
			HOFFSET3D(resultrho.ptr, resultrho.pitch, nsdm, curmodel),
			resultrho.pitch,
			curmodelptr->dev_rho.ptr,
			curmodelptr->dev_rho.pitch,
			curmodelptr->dev_rho.xsize * sizeof(real),
			curmodelptr->dev_rho.ysize,
			cudaMemcpyDeviceToHost));

	}

#define WRITE_RESULT(x,y) if (currentConfig.y != NULL) writeMatrix(x, currentConfig.y);
	curdeviceptr = &gpus[0];
	CUDA_CHECK_RETURN(cudaSetDevice(curdeviceptr->realdevice));
	printf("Writing results\n");
	WRITE_RESULT(resultW, weightsfile)
	WRITE_RESULT(resultA, afile)
	WRITE_RESULT(resultc, cfile)
	WRITE_RESULT(resultLtall, ltallfile)
	WRITE_RESULT(resultgm, gmfile)
	WRITE_RESULT(resultalpha, alphafile)
	WRITE_RESULT(resultbeta, betafile)
	WRITE_RESULT(resultmu, mufile)
	WRITE_RESULT(resultrho, rhofile)

	if (currentConfig.llfile != NULL) {
		cudaPitchedPtr host_ll_ptr;
		host_ll_ptr.ptr = host_ll;
		host_ll_ptr.xsize = maxiter;
		host_ll_ptr.pitch = maxiter * sizeof(real);
		host_ll_ptr.ysize = 1;
		writeMatrix(host_ll_ptr, currentConfig.llfile);
	}
	if (currentConfig.spherefile != NULL) {
		writeDevMatrix(curdeviceptr->sphere, currentConfig.spherefile);
	}

	/*
	 * Destroy streams and free memory
	 */

	C_CHECK_RETURN(finalizeModels());
	C_CHECK_RETURN(finalizeDevs());

	free(host_ll);

	CUDA_CHECK_RETURN(cudaFreeHost(resultA.ptr));
	CUDA_CHECK_RETURN(cudaFreeHost(resultW.ptr));
	CUDA_CHECK_RETURN(cudaFreeHost(resultc.ptr));
	CUDA_CHECK_RETURN(cudaFreeHost(resultgm.ptr));
	CUDA_CHECK_RETURN(cudaFreeHost(resultLtall.ptr));
	CUDA_CHECK_RETURN(cudaFreeHost(resultalpha.ptr));
	CUDA_CHECK_RETURN(cudaFreeHost(resultbeta.ptr));
	CUDA_CHECK_RETURN(cudaFreeHost(resultmu.ptr));
	CUDA_CHECK_RETURN(cudaFreeHost(resultrho.ptr));



	printf("Done!\n");
	return SUCCESS;


}
