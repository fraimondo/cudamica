/*
 * helpers.cu
 *
 *  Created on: Jul 24, 2012
 *      Author: fraimondo
 */


#include <helpers.h>
#include <tools.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


//TODO: Maiximize blocksize according to available mem
error recalc_config() {
	DPRINTF(1,"Recalculating blocks\n");
	natural nchannels = currentConfig.nchannels;
	natural nsamples = currentConfig.nsamples;
	natural nmmodels = currentConfig.nmmodels;
	natural nsdm = currentConfig.nsdm;
	natural maxblocksize = currentConfig.block_size;
	natural ndevices = currentConfig.devcount;

	natural blockgpumodel = nsamples/(maxblocksize*ndevices);

	while (nsamples < blockgpumodel * ndevices * maxblocksize) {
		maxblocksize--;
		blockgpumodel = nsamples/(maxblocksize*ndevices);
	}

	natural dropped = nsamples - (blockgpumodel * ndevices * maxblocksize);
	while (dropped != 0) {
		maxblocksize--;
		blockgpumodel = nsamples/(maxblocksize*ndevices);
		dropped = nsamples - (blockgpumodel * ndevices * maxblocksize);
	}

	// TODO: check if block_size is a good value
	DPRINTF(1, "Recalculated block size: %lu (%lu dropped) %lu per gpu per model\n", maxblocksize, dropped, blockgpumodel);

	currentConfig.block_size = maxblocksize;
	currentConfig.blocks_per_gpu_per_model = blockgpumodel;

	return SUCCESS;
}


/*
 * Tune for device i;
 */
error autotune(natural device) {

	natural nchannels = currentConfig.nchannels;
	natural nsamples = currentConfig.nsamples;
	natural nmmodels = currentConfig.nmmodels;
	natural nsdm = currentConfig.nsdm;
	natural warpsize = gpus[device].deviceProp->warpSize;
	natural block_size = currentConfig.block_size;

	devicetune * dimensions = &gpus[device].dimensions;

	natural maxshared = gpus[device].deviceProp->sharedMemPerBlock;

	natural maxthreads = gpus[device].deviceProp->maxThreadsPerBlock;
	natural minblocks = gpus[device].deviceProp->multiProcessorCount * 8;

	natural threadx;
	natural thready;
	natural blockx;
	natural blocky;


	/*
	 * Each thread process single data
	 * Each block process N samples
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension
	 */

	threadx = nchannels;
	thready = nsamples/minblocks;

	while (thready*threadx > maxthreads) {
		thready --;
	}

	while (nsamples % thready != 0) {
		thready --;
	}

	dimensions->c1c1s_12dim.nthreads = dim3(threadx, thready, 1);
	dimensions->c1c1s_12dim.nblocks = dim3(nsamples/thready, 1, 1);
	dimensions->c1c1s_12dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (12dim): Using a (%d, 1) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->c1c1s_12dim.nblocks.x,
		dimensions->c1c1s_12dim.nthreads.x,
		dimensions->c1c1s_12dim.nthreads.y,
		dimensions->c1c1s_12dim.nthreads.y * dimensions->c1c1s_12dim.nblocks.x,
		nsamples - (dimensions->c1c1s_12dim.nblocks.x*dimensions->c1c1s_12dim.nthreads.y));

	/*
	 * Each thread process M sample
	 * Each block process N samples
	 * blockDim has 1 dimension
	 * gridDim has 1 dimension
	 */
	blockx = minblocks;
	while (nsamples % blockx != 0) {
		blockx ++;
	}

	dimensions->c1cNs_11dim.nthreads = dim3(nchannels, 1, 1);
	dimensions->c1cNs_11dim.nblocks = dim3(blockx, 1, 1);
	dimensions->c1cNs_11dim.ndata = nsamples/dimensions->c1cNs_11dim.nblocks.x;

	DPRINTF(1,
		"N Data configuration (11dim): Using a (%d, 1) grid of (%d, %d) threads to process %d samples (%lu per block, %lu dropped)\n",
		dimensions->c1cNs_11dim.nblocks.x,
		dimensions->c1cNs_11dim.nthreads.x,
		dimensions->c1cNs_11dim.nthreads.y,
		dimensions->c1cNs_11dim.nthreads.y * dimensions->c1cNs_11dim.nblocks.x,
		dimensions->c1cNs_11dim.ndata,
		nsamples % (dimensions->c1cNs_11dim.nblocks.x*dimensions->c1cNs_11dim.nthreads.y));




	/* Channels by Channels
	 * Each thread process 1 element
	 * Each block process 1 row
	 * blockDim has 1 dimension
	 * gridDim has 1 dimension
	 */
	dimensions->c1c1c_11dim.nthreads = dim3(nchannels, 1, 1);
	dimensions->c1c1c_11dim.nblocks = dim3(nchannels, 1, 1);
	dimensions->c1c1c_11dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (11dim - by nchannels): Using a (%d, 1) grid of (%d, %d) threads to process %d channels (%lu dropped)\n",
		dimensions->c1c1c_11dim.nblocks.x,
		dimensions->c1c1c_11dim.nthreads.x,
		dimensions->c1c1c_11dim.nthreads.y,
		dimensions->c1c1c_11dim.nthreads.y * dimensions->c1c1c_11dim.nblocks.x,
		nchannels - (dimensions->c1c1c_11dim.nblocks.x*dimensions->c1c1c_11dim.nthreads.y));

	/*
	 * Each thread process 1 element
	 * Each block process nthreads elements
	 * blockDim has 2 dimension by nmmodels
	 * gridDim has 1 dimension
	 */

	blockx = minblocks;
	// cannot be bigger than maxthreads
	while ((block_size / blockx) *2> maxthreads) {
		blockx++;
	}

	natural thisblockmem = block_size/blockx * (nmmodels+1) * sizeof(real);
	while (thisblockmem > maxshared) {
		thisblockmem = block_size/blockx * (nmmodels+1) * sizeof(real);
		blockx++;
	}

	// must divide block size
	while (block_size % blockx != 0) {
		blockx++;
	}



	dimensions->blocksize_12dim.nthreads = dim3(block_size/blockx, nmmodels, 1);
	dimensions->blocksize_12dim.nblocks = dim3(blockx, 1, 1);
	dimensions->blocksize_12dim.ndata = 1;
	DPRINTF(1,
		"Single Data vector configuration (12dim - block_size by nmmodels): Using a (%d, 1) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->blocksize_12dim.nblocks.x,
		dimensions->blocksize_12dim.nthreads.x,
		dimensions->blocksize_12dim.nthreads.y,
		dimensions->blocksize_12dim.nthreads.x * dimensions->blocksize_12dim.nblocks.x,
		block_size - (dimensions->blocksize_12dim.nblocks.x*dimensions->blocksize_12dim.nthreads.x));

	/*
	 * Each thread process 1 element
	 * Each block process nthreads elements
	 * blockDim has 1 dimension
	 * gridDim has 1 dimension
	 */

	blockx = minblocks;
	// cannot be bigger than maxthreads
	while (block_size / blockx > maxthreads) {
		blockx++;
	}

	while (block_size % blockx != 0) {
		blockx++;
	}


	dimensions->vblocksize_11dim.nthreads = dim3(block_size/blockx, 1, 1);
	dimensions->vblocksize_11dim.nblocks = dim3(blockx, 1, 1);
	dimensions->vblocksize_11dim.ndata = 1;
	DPRINTF(1,
		"Single Data vector configuration (11dim - block_size): Using a (%d, 1) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->vblocksize_11dim.nblocks.x,
		dimensions->vblocksize_11dim.nthreads.x,
		dimensions->vblocksize_11dim.nthreads.y,
		dimensions->vblocksize_11dim.nthreads.x * dimensions->vblocksize_11dim.nblocks.x,
		block_size - (dimensions->vblocksize_11dim.nblocks.x*dimensions->vblocksize_11dim.nthreads.x));

	dimensions->vchannels_11dim.nthreads = dim3(nchannels, 1, 1);
	dimensions->vchannels_11dim.nblocks = dim3(1, 1, 1);
	dimensions->vchannels_11dim.ndata = 1;
	DPRINTF(1,
		"Single Data vector configuration (11dim - nchannels): Using a (%d, 1) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->vchannels_11dim.nblocks.x,
		dimensions->vchannels_11dim.nthreads.x,
		dimensions->vchannels_11dim.nthreads.y,
		dimensions->vchannels_11dim.nthreads.x * dimensions->vchannels_11dim.nblocks.x,
		nchannels - (dimensions->vchannels_11dim.nblocks.x*dimensions->vchannels_11dim.nthreads.x));

	/*
	 * Each thread process single data
	 * Each block process N samples
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension
	 */

	threadx = nchannels;
	thready = block_size/minblocks;

	while (thready*threadx > maxthreads) {
		thready --;
	}

	while (block_size % thready != 0) {
		thready --;
	}

	dimensions->c1c1bs_12dim.nthreads = dim3(threadx, thready, 1);
	dimensions->c1c1bs_12dim.nblocks = dim3(block_size/thready, 1, 1);
	dimensions->c1c1bs_12dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (12dim - blocksize): Using a (%d, 1) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->c1c1bs_12dim.nblocks.x,
		dimensions->c1c1bs_12dim.nthreads.x,
		dimensions->c1c1bs_12dim.nthreads.y,
		dimensions->c1c1bs_12dim.nthreads.y * dimensions->c1c1bs_12dim.nblocks.x,
		block_size - (dimensions->c1c1bs_12dim.nblocks.x*dimensions->c1c1bs_12dim.nthreads.y));


	/*
	 * Each thread process multiple data
	 * Each block process N channels
	 * blockDim has 1 dimensions
	 * gridDim has 2 dimension.
	 * nsdm = blockIdx.y
	 */
	threadx = nchannels;
	thready = 1;

	blockx = minblocks;
	while (block_size % blockx != 0) {
		blockx ++;
	}

	dimensions->c1cNbs_21dim.nthreads = dim3(threadx, thready, 1);
	dimensions->c1cNbs_21dim.nblocks = dim3(blockx, nsdm, 1);
	dimensions->c1cNbs_21dim.ndata = block_size/blockx;
	DPRINTF(1,
		"N Data configuration (21dim - blocksize): Using a (%d, %d) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->c1cNbs_21dim.nblocks.x,
		dimensions->c1cNbs_21dim.nblocks.y,
		dimensions->c1cNbs_21dim.nthreads.x,
		dimensions->c1cNbs_21dim.nthreads.y,
		dimensions->c1cNbs_21dim.nblocks.x * dimensions->c1cNbs_21dim.ndata,
		block_size - (dimensions->c1cNbs_21dim.nblocks.x* dimensions->c1cNbs_21dim.ndata));

	/*
	 * Each thread process multiple data
	 * Each block process N channels
	 * blockDim has 1 dimensions
	 * gridDim has 1 dimension.
	 */
	threadx = nchannels;
	thready = 1;

	blockx = minblocks;
	while (block_size % blockx != 0) {
		blockx ++;
	}

	dimensions->c1cNbs_11dim.nthreads = dim3(threadx, thready, 1);
	dimensions->c1cNbs_11dim.nblocks = dim3(blockx, 1, 1);
	dimensions->c1cNbs_11dim.ndata = block_size/blockx;
	DPRINTF(1,
		"N Data configuration (11dim - blocksize): Using a (%d, %d) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->c1cNbs_11dim.nblocks.x,
		dimensions->c1cNbs_11dim.nblocks.y,
		dimensions->c1cNbs_11dim.nthreads.x,
		dimensions->c1cNbs_11dim.nthreads.y,
		dimensions->c1cNbs_11dim.nblocks.x * dimensions->c1cNbs_11dim.ndata,
		block_size - (dimensions->c1cNbs_11dim.nblocks.x* dimensions->c1cNbs_11dim.ndata));


	/*
	 * Each thread process single data
	 * Each block process N samples
	 * blockDim has 2 dimensions
	 * gridDim has 2 dimension.
	 * nsdm = blockIdx.y
	 */

	dimensions->c1c1snsdm_22dim.nthreads = dim3(threadx, thready, 1);
	dimensions->c1c1snsdm_22dim.nblocks = dim3(block_size/thready, nsdm, 1);
	dimensions->c1c1snsdm_22dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (22dim - by nsdm): Using a (%d, %d) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->c1c1snsdm_22dim.nblocks.x,
		dimensions->c1c1snsdm_22dim.nblocks.y,
		dimensions->c1c1snsdm_22dim.nthreads.x,
		dimensions->c1c1snsdm_22dim.nthreads.y,
		dimensions->c1c1snsdm_22dim.nthreads.y * dimensions->c1c1snsdm_22dim.nblocks.x,
		block_size - (dimensions->c1c1snsdm_22dim.nblocks.x*dimensions->c1c1snsdm_22dim.nthreads.y));

	/*
	 * Each thread process 1 sample
	 * Each block process N samples
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * nsdm = threadIdx.y
	 */
	dimensions->c1c1snsdm_12dim.nthreads = dim3(nchannels, nsdm, 1);
	dimensions->c1c1snsdm_12dim.nblocks = dim3(block_size, 1, 1);
	dimensions->c1c1snsdm_12dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (12dim - by nsdm): Using a (%d, %d) grid of (%d, %d) threads to process %d samples (%lu dropped)\n",
		dimensions->c1c1snsdm_12dim.nblocks.x,
		dimensions->c1c1snsdm_12dim.nblocks.y,
		dimensions->c1c1snsdm_12dim.nthreads.x,
		dimensions->c1c1snsdm_12dim.nthreads.y,
		dimensions->c1c1snsdm_12dim.nblocks.x,
		block_size - (dimensions->c1c1snsdm_12dim.nblocks.x));

	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.t = nsdm * nmmodels
	 */
	threadx = nchannels;
	natural rows = nsdm * nmmodels;

	if (minblocks > rows) { //less rows than blocks... one row per block
		thready = 1;
		blockx = rows;
	} else { //more rows than blocks.. try to split.
		blockx = minblocks;
		thready = rows /blockx;
		// cannot be bigger than maxthreads
		while (thready * threadx > maxthreads) {
			blockx++;
			thready = rows / blockx;
		}
		while (thready % blockx != 0) {
			blockx++;
			thready = nsdm * nmmodels / blockx;
		}
	}





	dimensions->c1c1snsdmnmmodels_12dim.nthreads = dim3(threadx, thready, 1);
	dimensions->c1c1snsdmnmmodels_12dim.nblocks = dim3(blockx, 1, 1);
	dimensions->c1c1snsdmnmmodels_12dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (12dim - chans by nsdm by nmmodels): Using a (%d, %d) grid of (%d, %d) threads to process %d rows (%lu dropped)\n",
		dimensions->c1c1snsdmnmmodels_12dim.nblocks.x,
		dimensions->c1c1snsdmnmmodels_12dim.nblocks.y,
		dimensions->c1c1snsdmnmmodels_12dim.nthreads.x,
		dimensions->c1c1snsdmnmmodels_12dim.nthreads.y,
		dimensions->c1c1snsdmnmmodels_12dim.nblocks.x * dimensions->c1c1snsdmnmmodels_12dim.nthreads.y,
		nsdm * nmmodels - (dimensions->c1c1snsdmnmmodels_12dim.nblocks.x * dimensions->c1c1snsdmnmmodels_12dim.nthreads.y));

	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.t = nmmodels
	 */
	threadx = nchannels;
	rows =  nmmodels;

	if (minblocks > rows) { //less rows than blocks... one row per block
		thready = 1;
		blockx = rows;
	} else { //more rows than blocks.. try to split.
		blockx = minblocks;
		thready = rows /blockx;
		// cannot be bigger than maxthreads
		while (thready * threadx > maxthreads) {
			blockx++;
			thready = rows / blockx;
		}
		while (thready % blockx != 0) {
			blockx++;
			thready = nmmodels / blockx;
		}
	}

	dimensions->c1c1snmmodels_12dim.nthreads = dim3(threadx, thready, 1);
	dimensions->c1c1snmmodels_12dim.nblocks = dim3(blockx, 1, 1);
	dimensions->c1c1snmmodels_12dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (12dim - chans by nmmodels): Using a (%d, %d) grid of (%d, %d) threads to process %d rows (%lu dropped)\n",
		dimensions->c1c1snmmodels_12dim.nblocks.x,
		dimensions->c1c1snmmodels_12dim.nblocks.y,
		dimensions->c1c1snmmodels_12dim.nthreads.x,
		dimensions->c1c1snmmodels_12dim.nthreads.y,
		dimensions->c1c1snmmodels_12dim.nblocks.x * dimensions->c1c1snmmodels_12dim.nthreads.y,
		nmmodels - (dimensions->c1c1snmmodels_12dim.nblocks.x * dimensions->c1c1snmmodels_12dim.nthreads.y));

	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.t = nmmodels
	 */
	threadx = nchannels;
	blockx =  nmmodels;


	dimensions->c1c1snmmodels_11dim.nthreads = dim3(threadx, 1, 1);
	dimensions->c1c1snmmodels_11dim.nblocks = dim3(blockx, 1, 1);
	dimensions->c1c1snmmodels_11dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (11dim - chans by nmmodels): Using a (%d, %d) grid of (%d, %d) threads to process %d rows (%lu dropped)\n",
		dimensions->c1c1snmmodels_11dim.nblocks.x,
		dimensions->c1c1snmmodels_11dim.nblocks.y,
		dimensions->c1c1snmmodels_11dim.nthreads.x,
		dimensions->c1c1snmmodels_11dim.nthreads.y,
		dimensions->c1c1snmmodels_11dim.nblocks.x * dimensions->c1c1snmmodels_11dim.nthreads.y,
		nmmodels - (dimensions->c1c1snmmodels_11dim.nblocks.x * dimensions->c1c1snmmodels_11dim.nthreads.y));


	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.y = nsdm
	 */
	threadx = nchannels;
	blockx =  nsdm;


	dimensions->c1c1snsdm_11dim.nthreads = dim3(threadx, 1, 1);
	dimensions->c1c1snsdm_11dim.nblocks = dim3(blockx, 1, 1);
	dimensions->c1c1snsdm_11dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (11dim - chans by nsdm): Using a (%d, %d) grid of (%d, %d) threads to process %d rows (%lu dropped)\n",
		dimensions->c1c1snsdm_11dim.nblocks.x,
		dimensions->c1c1snsdm_11dim.nblocks.y,
		dimensions->c1c1snsdm_11dim.nthreads.x,
		dimensions->c1c1snsdm_11dim.nthreads.y,
		dimensions->c1c1snsdm_11dim.nblocks.x * dimensions->c1c1snsdm_11dim.nthreads.y,
		nsdm - (dimensions->c1c1snsdm_11dim.nblocks.x * dimensions->c1c1snsdm_11dim.nthreads.y));



	/*
	 * Each thread.x process 1 element
	 * Each thread.y process 1 row
	 * Each block process N rows
	 * blockDim has 2 dimensions
	 * gridDim has 1 dimension.
	 * block.x * thread.t = nsdm * nmmodels
	 */
	threadx = nchannels;
	blockx = nchannels;
	thready = nmmodels;

	dimensions->c1c1snchansnmmodels_12dim.nthreads = dim3(threadx, thready, 1);
	dimensions->c1c1snchansnmmodels_12dim.nblocks = dim3(blockx, 1, 1);
	dimensions->c1c1snchansnmmodels_12dim.ndata = 1;
	DPRINTF(1,
		"Single Data configuration (12dim - chans by chans by nmmodels): Using a (%d, %d) grid of (%d, %d) threads to process %d rows (%lu dropped)\n",
		dimensions->c1c1snchansnmmodels_12dim.nblocks.x,
		dimensions->c1c1snchansnmmodels_12dim.nblocks.y,
		dimensions->c1c1snchansnmmodels_12dim.nthreads.x,
		dimensions->c1c1snchansnmmodels_12dim.nthreads.y,
		dimensions->c1c1snchansnmmodels_12dim.nblocks.x * dimensions->c1c1snchansnmmodels_12dim.nthreads.y,
		nchannels * nmmodels - (dimensions->c1c1snchansnmmodels_12dim.nblocks.x * dimensions->c1c1snchansnmmodels_12dim.nthreads.y));


	return SUCCESS;
}

error loadMatrix(real * data, natural rows, natural elems, char* filename, bool isdouble) {
	int fd = open(filename, O_RDONLY);
	if (fd == 0) return ERRORNOFILE;
	struct stat sb;
	if (fstat(fd, &sb) == -1) {
		fprintf(stderr, "Error stating data file %s\n", filename);
		return ERRORNOFILE;
	}
	void *mmaping = mmap (0, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
	size_t map_size = sb.st_size;
	if (! isdouble) {
		float *matriz = (float*)mmaping;
		if (matriz == MAP_FAILED) {
			fprintf(stderr, "Error mapping data file %s\n", filename);
			return ERRORNOFILE;
		}
		DPRINTF(2, "Matrix mapped at %p\n", matriz);
		DPRINTF(2, "loadMatrix from %p (%lu x %lu) to %p\n", matriz, rows, elems, data);
		for (int i = 0; i < rows*elems; i++) {
			data[i] = (real)matriz[i];
		}
	} else {
		double *matriz = (double*)mmaping;
		if (matriz == MAP_FAILED) {
			fprintf(stderr, "Error mapping data file %s\n", filename);
			return ERRORNOFILE;
		}
		DPRINTF(2, "Matrix mapped at %p\n", matriz);
		DPRINTF(2, "loadMatrix from %p (%lu x %lu) to %p\n", matriz, rows, elems, data);
		for (int i = 0; i < rows*elems; i++) {
			data[i] = (real)matriz[i];
		}
	}
	if (close(fd) == -1) {
		fprintf(stderr, "Error closing data file %d\n",fd);
	}
	if (munmap (mmaping, map_size) == -1) {
		fprintf(stderr, "Error unmapping data file at %p with size %lu\n", mmaping, map_size);
	}
	return SUCCESS;
}

error writeValue(real value, char * filename) {
	FILE * output = fopen(filename, "w");
	DPRINTF(1, "Writing value %.16f  to %s\n", value, filename);
	if (output == NULL) return ERRORNOFILE;
	error retorno;
	if (fwrite((void*)&value, sizeof(real), 1, output) != 1) {
		retorno = ERRORIO;
	} else {
		retorno = SUCCESS;
	}
	fclose(output);
	return retorno;
}

error writeMatrix(cudaPitchedPtr ptr, char * filename) {
	FILE * output = fopen(filename, "w");
	DPRINTF(1, "Writing matrix at %p (%lu by %lu) to %s\n",ptr.ptr, ptr.xsize, ptr.ysize, filename);
	if (output == NULL) return ERRORNOFILE;
	real* data = (real*)ptr.ptr;
	natural realelems = ptr.pitch/sizeof(real);
	natural elems = ptr.xsize;
	natural rows = ptr.ysize;

	error retorno = SUCCESS;
	for (int i = 0; i < rows; i++) {
		if (fwrite((void*)&data[i * realelems], sizeof(real), elems, output) != elems ) {
			retorno = ERRORIO;
			break;
		}
	}

	fclose(output);
	return retorno;
}

error writeDevMatrix(cudaPitchedPtr ptr, char * filename) {
	FILE * output = fopen(filename, "w");
	DPRINTF(1, "Writing matrix at %p (%lu by %lu) to %s\n",ptr.ptr, ptr.xsize, ptr.ysize, filename);
	if (output == NULL) return ERRORNOFILE;
	real * temp;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaHostAlloc((void **)&temp, ptr.ysize * sizeof(real) * ptr.xsize, cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaMemcpy2D(temp, sizeof(real) * ptr.xsize, ptr.ptr, ptr.pitch, ptr.xsize * sizeof(real), ptr.ysize, cudaMemcpyDeviceToHost));
	error retorno = SUCCESS;
	int realelems = ptr.xsize;
	for (int i = 0; i < ptr.ysize; i++) {
		if (fwrite((void*)&temp[i * realelems], sizeof(real), ptr.xsize, output) != ptr.xsize ) {
			retorno = ERRORIO;
			break;
		}
	}
	CUDA_CHECK_RETURN(cudaFreeHost(temp));
	fclose(output);
	return retorno;
}

error writeDevMatrix3d(cudaPitchedPtr ptr, natural zsize, char * filename) {
	FILE * output = fopen(filename, "w");
	DPRINTF(1, "Writing matrix at %p (%lu by %lu) to %s\n",ptr.ptr, ptr.xsize, ptr.ysize, filename);
	if (output == NULL) return ERRORNOFILE;
	real * temp;
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMallocHost((void **)&temp, ptr.ysize * sizeof(real) * ptr.xsize * zsize));
	CUDA_CHECK_RETURN(cudaMemcpy2D(temp, sizeof(real) * ptr.xsize, ptr.ptr, ptr.pitch, ptr.xsize * sizeof(real), ptr.ysize * zsize, cudaMemcpyDeviceToHost));
	error retorno = SUCCESS;
	int realelems = ptr.xsize;
	for (int i = 0; i < ptr.ysize * zsize; i++) {
		if (fwrite((void*)&temp[i * realelems], sizeof(real), ptr.xsize, output) != ptr.xsize ) {
			retorno = ERRORIO;
			break;
		}
	}
	CUDA_CHECK_RETURN(cudaFreeHost(temp));
	fclose(output);
	return retorno;
}

error loadDevMatrix(cudaPitchedPtr ptr, char* filename, bool isdouble) {
	int fd = open(filename, O_RDONLY);
	if (fd == 0) return ERRORNOFILE;
	struct stat sb;
	if (fstat(fd, &sb) == -1) {
		fprintf(stderr, "Error stating data file %s\n", filename);
		return ERRORNOFILE;
	}
	void *mmaping = mmap (0, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
	size_t map_size = sb.st_size;
	real * data = NULL;
	if (!isdouble) {
		float *matriz = (float*)mmaping;
		if (matriz == MAP_FAILED) {
			fprintf(stderr, "Error mapping data file %s\n", filename);
			return ERRORNOFILE;
		}
		DPRINTF(2, "Matrix mapped at %p\n", matriz);
		natural rows = ptr.ysize;
		natural elems = ptr.xsize;
		 data = (real *)malloc(rows*elems*sizeof(real));
		if (data == NULL) {
			return ERRORNOMEM;
		}
		for (int i = 0; i < rows*elems; i++) {
			data[i] = (real)matriz[i];
		}
		DPRINTF(2, "loadDevMatrix from %p (%lu x %lu) to %p\n", matriz, ptr.ysize, ptr.xsize, ptr.ptr);
	} else {
		double *matriz = (double*)mmaping;
		if (matriz == MAP_FAILED) {
			fprintf(stderr, "Error mapping data file %s\n", filename);
			return ERRORNOFILE;
		}
		DPRINTF(2, "Matrix mapped at %p\n", matriz);
		natural rows = ptr.ysize;
		natural elems = ptr.xsize;
		data = (real *)malloc(rows*elems*sizeof(real));
		if (data == NULL) {
			return ERRORNOMEM;
		}
		for (int i = 0; i < rows*elems; i++) {
			data[i] = (real)matriz[i];
		}
		DPRINTF(2, "loadDevMatrix from %p (%lu x %lu) to %p\n", matriz, ptr.ysize, ptr.xsize, ptr.ptr);
	}

	CUDA_CHECK_RETURN(cudaMemcpy2D(
			ptr.ptr,
			ptr.pitch,
			(void *)data,
			ptr.xsize * sizeof(real),
			ptr.xsize * sizeof(real),
			ptr.ysize,
			cudaMemcpyHostToDevice
		));

	free(data);

	if (close(fd) == -1) {
		fprintf(stderr, "Error closing data file %d\n",fd);
	}
	if (munmap (mmaping, map_size) == -1) {
		fprintf(stderr, "Error unmapping data file at %p with size %lu\n", mmaping, map_size);
	}
	return SUCCESS;
}

error loadDevMatrix3d(cudaPitchedPtr ptr, natural index, char* filename) {
	int fd = open(filename, O_RDONLY);
	if (fd == 0) return ERRORNOFILE;
	struct stat sb;
	if (fstat(fd, &sb) == -1) {
		fprintf(stderr, "Error stating data file %s\n", filename);
		return ERRORNOFILE;
	}
	void *mmaping = mmap (0, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
	size_t map_size = sb.st_size;
	float *matriz = (float*)mmaping;
	if (matriz == MAP_FAILED) {
		fprintf(stderr, "Error mapping data file %s\n", filename);
		return ERRORNOFILE;
	}

	DPRINTF(2, "Matrix mapped at %p\n", matriz);
	natural rows = ptr.ysize;
	natural elems = ptr.xsize;
	real * data = (real *)malloc(rows*elems*sizeof(real));
	if (data == NULL) {
		return ERRORNOMEM;
	}
	matriz += index * rows * elems;
	for (int i = 0; i < rows*elems; i++) {
		data[i] = (real)matriz[i];
	}
	DPRINTF(2, "loadDevMatrix from %p (%lu x %lu) to %p\n", matriz, ptr.ysize, ptr.xsize, ptr.ptr);
	CUDA_CHECK_RETURN(cudaMemcpy2D(
			ptr.ptr,
			ptr.pitch,
			(void *)data,
			ptr.xsize * sizeof(real),
			ptr.xsize * sizeof(real),
			ptr.ysize,
			cudaMemcpyHostToDevice
		));

	free(data);

	if (close(fd) == -1) {
		fprintf(stderr, "Error closing data file %d\n",fd);
	}
	if (munmap (mmaping, map_size) == -1) {
		fprintf(stderr, "Error unmapping data file at %p with size %lu\n", mmaping, map_size);
	}
	return SUCCESS;
}
