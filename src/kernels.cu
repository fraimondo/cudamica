#include <kernels.h>
#include <tipos.h>
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
 * Zero a matrix
 */
__global__ void zeros(real * a, natural elemsperrow) {
	a[blockIdx.x * elemsperrow + threadIdx.x] = 0.0;
}

/*
 * Initializes a vector in a constant value
 * Vector dimensions are (nthreads)
 */
__global__ void constant1D(real * a, real val) {
	a[blockIdx.x * blockDim.x + threadIdx.x] = val;
}


/*
 * Initializes a matrix in a constant value
 * Matrix dimensions are (nblocks, nthreads)
 */
__global__ void constant2D(real * a, natural elemsperrow, real val) {
	a[blockIdx.x * elemsperrow + threadIdx.x] = val;
}

/*
 * Initializes a matrix in a constant value
 * Matrix dimensions are (Y= nblocks.x, Z= nthreads.y, X=nthreads.x)
 */
__global__ void constant3D(real * a, natural elemsperrow, real val) {
	natural sample = blockIdx.x + blockDim.x * blockIdx.y;
	natural zoffset = threadIdx.y * gridDim.x * gridDim.y * elemsperrow;
	natural yoffset = sample * elemsperrow;
	a[zoffset + yoffset + threadIdx.x] = val;
}



/*
 * Computes the sums for each channel and then divides by the number of samples.
 * sums = sum(channels(data))/samples
 *
 * Should be launched with N blocks of channels threads
 *
 * data: matrix
 * channels: number of channels
 * samples: number of samples
 * pitch: matrix row size in bytes
 * sums: output matrix (must be at least blocks by channels)
 * sumspitch: sums row size in bytes
 */
__global__ void getMean(real* data, natural nchannels, natural nsamples, natural nsamplesperblock, natural drowsize, real* sums, natural sumsrowsize) {
	float sum = 0.0;
	__shared__ bool isLastBlockFinished; //If true, then last block has finished.
	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts
	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
		end = nsamples;
	}
	for (; i < end; i++) {
		sum += data[(i*drowsize) + threadIdx.x];
	}
	sums[blockIdx.x * sumsrowsize + threadIdx.x] = sum;

	if (threadIdx.x == 0) {
		natural value = atomicInc(&blocksFinished, gridDim.x);
		isLastBlockFinished = (value == gridDim.x-1);
	}

	__syncthreads();
	if (isLastBlockFinished) {
		sum = 0.0;
		for (i = 0; i < gridDim.x; i++) {
			sum += sums[threadIdx.x + i * sumsrowsize];
		}
		sums[threadIdx.x] = sum/nsamples;
		if (threadIdx.x == 0) {
			blocksFinished = 0;
		}
	}
}
/*
 * Centers data by substracting the mean value from means vector
 * data = data - mean
 *
 * Should be launched with N blocks of channels by (X * M) threads where X * M = samples
 *
 * data: matrix
 * colwidth: real elements per row size
 * means: vector of means
 */
__global__ void subMean(real* data, natural colwidth, const real* means) {
	real mean = means[threadIdx.x];
	int offset = blockIdx.x * blockDim.y + threadIdx.y;
	data[(offset*colwidth) + threadIdx.x] -= mean;
}

/*
 * Sets a matrix to be the eye matrix.
  *
 * Should be launched with N blocks of rows by M threads of channels
 *
 * data: matrix
 * colwidth: real elements per row size
 */
__global__ void eye(real* data, natural colwidth) {
	int offset = blockIdx.x * colwidth;
	real value = blockIdx.x == threadIdx.x ? 1.0 : 0.0;
	data[offset + threadIdx.x] = value;
}


/*
 * Scales a matrix by scalar
 *
 * Should be launched with N blocks of rows by M threads of columns
 *
 * data: matrix
 * colwidth: real elements per row size
 * scalar: real value to multiply
 */
__global__ void scale(real* data, natural colwidth, real scalar) {
	int offset = blockIdx.x * colwidth;
	data[offset + threadIdx.x] *= scalar;
}


/*
 * Computes variances of data.
 * Should be launched with a single block of channel threads
 *
 * From: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 *
 */
__global__ void getvariance(real * data, natural delemsperrow, natural nsamples, real * sphere, natural selemsperrow, real * means) {
	real mean 	= means[threadIdx.x];
	real M2		= 0.0;
	real delta 	= 0.0;
	real value 	= 0.0;

	for (int n = 0; n < nsamples; n++) {
		value = data[(n*delemsperrow) + threadIdx.x];
		delta = abs(value - mean);
		M2 = M2 + delta * delta;
	}

	sphere[threadIdx.x + selemsperrow * threadIdx.x] = 1/sqrt(M2/(nsamples-1));
}




/*
 * Add eye to matrix
 *
 */
__global__ void addEye(real * a, natural elemsperrow) {
	if (blockIdx.x == threadIdx.x) {
		a[blockIdx.x * elemsperrow + threadIdx.x] += 1.0;
	}
}


/*
 * Multiply and add constants
 * a = matrix(C,R)
 * Should be launched with (a, 1) blocks of (b, c) threads with C = b and R = a * c
 */
__global__ void mpaddConstants(real * a, natural elemsperrow, real mult, real add) {
	a[blockIdx.x * elemsperrow + threadIdx.x] =
			(a[blockIdx.x * elemsperrow+ threadIdx.x] + add) * mult;
}

/*
 * Multiply each element in the diagonal
 */
__global__ void getDiagonalMult(real *a, natural elemsperrow, natural rows) {
	real value = 1.0;
	for (int i = 0; i < rows; i++) {
		value *= a[i + elemsperrow * i];
	}
	a[0] = value;
}

/*
 * Normalize A by channel
 *
 * Should be launched with nchannels blocks of nchannels threads;
 * Shared mem = nchannels * sizeof(real)
 */
extern __shared__ real column[];
__global__ void normalize(real * a, natural elemsperrow) {
	real data = a[blockIdx.x * elemsperrow + threadIdx.x];
	column[threadIdx.x] = data * data;
	__syncthreads();
	if (threadIdx.x == 0) {
		real sum = column[0];
		for (int i = 1; i < blockDim.x; i++) {
			sum += column[i];
		}
		column[0] = sqrt(sum);
	}
	__syncthreads();
	real norm = column[0];
	a[blockIdx.x * elemsperrow + threadIdx.x] = data/norm;
}

/*
 * Substract value to each sample acording to the
 * corresponding element in the vector
 * Should be launched with 2 dimensions blocks. One thread.y for each channel.
 */
extern __shared__ real vector[];
__global__ void substract(real *a, natural elemsperrow, real * values) {
	real value = 0.0;
	if (threadIdx.y == 0) {
		value = values[threadIdx.x];
		vector[threadIdx.x] = value;
	}
	__syncthreads();
	value = vector[threadIdx.x];
	natural offset = elemsperrow * (threadIdx.y + blockIdx.x * blockDim.y);
	a[offset + threadIdx.x] -= value;
}



/*
 * y = sqrt(beta) * (b - mu)
 * Perform the previous function for each sample in b
 * beta is beta[n] and mu is mu[n] where n is the channel value;
 * Depending on the channel, the sample is multiplied and substracted the coresponding
 * beta and mu value.
 * Should be launched with 2 dimensions blocks. One thread.x for each channel.
 * Should be launched with 2 dimensions grids. One block.y for each sdm
 */
extern __shared__ real vector[];
__global__ void betabyxminusmu(
		real *y, natural yelemsperrow,
		real * b, natural belemsperrow,
		real * betas, natural betaelemsperrow,
		real* mus, natural muelemsperrow,
		natural block_size) {
	real beta = 0.0;
	real mu = 0.0;
	real value = 0.0;
	if (threadIdx.y == 0) {
		beta = sqrt(betas[threadIdx.x + blockIdx.y * betaelemsperrow]);
		vector[threadIdx.x] = beta;
		mu = mus[threadIdx.x + blockIdx.y * muelemsperrow];
		vector[threadIdx.x + blockDim.x] = mu;
	}
	__syncthreads();
	beta = vector[threadIdx.x];
	mu = vector[threadIdx.x + blockDim.x];
	natural sample = (threadIdx.y + blockIdx.x * blockDim.y);
	value = beta * (b[sample * belemsperrow + threadIdx.x] - mu);
	y[(sample + block_size * blockIdx.y) * yelemsperrow + threadIdx.x] = value;
}


/*
 * 1 = log(alpha)  + 0.5 (log(beta)) - abs(y)^rho - log(2) - lgamma(1 + 1/rho)
 * Perform the previous function for each sample in y
 * beta is beta[n], alpha is alpha[n] and ro is ro[n] where n is the channel value;
 * Should be launched with 2 dimensions blocks. One thread.y for each channel.
 * Should be launched with 2 dimensions grids. One block.y for each sdm
 */
__global__ void logalogblogpfun(
		real * q, natural qelemsperrow,
		real * y, natural yelemsperrow,
		real * alphas, natural alphaelemsperrow,
		real * betas, natural betaelemsperrow,
		real * rhos, natural rhoelemsperrow,
		natural block_size) {
	real value = 0.0;
	real rho = 0.0;
	if (threadIdx.y == 0) {
		rho = rhos[threadIdx.x + blockIdx.y * rhoelemsperrow];
		value = log(alphas[threadIdx.x + blockIdx.y * alphaelemsperrow]) +
				0.5 * log(betas[threadIdx.x + blockIdx.y * betaelemsperrow]) -
				LOG2 - lgamma(1 + (1.0/rho));
		vector[threadIdx.x] = value;
		vector[threadIdx.x + blockDim.x] = rho;
	}
	__syncthreads();
	value = vector[threadIdx.x];
	rho = vector[threadIdx.x + blockDim.x];
	natural sample = (threadIdx.y + blockIdx.x * blockDim.y);
	natural sample3d = (sample + block_size * blockIdx.y);
	value = value - pow(abs(y[sample3d * yelemsperrow + threadIdx.x]), rho);
	//value = pow(abs(y[offset * yelemsperrow + threadIdx.x]), rho);
	q[sample3d * qelemsperrow + threadIdx.x] = value;
}


/*
 * Lt = Lt + Qmax + log(sum(exp(Q-Qmax)))
 * z = 1./exp(Qmax+log(sum(exp(Q-Qmax))))
 *
 * Should be launched with 2 dimensions per block. One thread.y for each sdm. One thread.x for each channel.
 * Shared mem = (nsdm+1) * nchannels * sizeof(real)
 */
__global__ void updateltz(
		real *q, natural qelemsperrow,
		real * lt, natural ltelemsperrow,
		real * z, natural zelemsperrow) {
	natural sample = blockIdx.x ;
	real qvalue =  q[threadIdx.x + (threadIdx.y * gridDim.x + sample)  * qelemsperrow];
	real value;
	real max = 0.0;
	real * sums = vector + blockDim.x;
	sums[threadIdx.x + threadIdx.y * blockDim.x] = qvalue;
	__syncthreads();
	if (threadIdx.y == 0) {
		max = qvalue;
		for (int i = 1; i < blockDim.y; i++) {
			if (max < sums[threadIdx.x + i *blockDim.x]) {
				max = sums[threadIdx.x + i *blockDim.x];
			}
		}
		vector[threadIdx.x] = max; //Qmax
	}

	__syncthreads();
	max = vector[threadIdx.x];
	sums[threadIdx.x + threadIdx.y * blockDim.x] = exp(qvalue - max); // exp(Q-Qmax)

	__syncthreads();
	if (threadIdx.y == 0) {
		value = 0.0;
		for (int i = 0; i < blockDim.y; i++) {
			value += sums[threadIdx.x + i * blockDim.x];
		}
		sums[threadIdx.x] = log(value); // qtmp
	}


	__syncthreads();
	z[(sample + threadIdx.y * gridDim.x) * zelemsperrow + threadIdx.x] =
			1.0/exp(vector[threadIdx.x] + sums[threadIdx.x] - qvalue);

//	z[(sample + threadIdx.y * gridDim.x) * zelemsperrow + threadIdx.x] = exp(qvalue - max);
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		value = 0.0;
		for (int i = 0; i < blockDim.x; i++) {
			value += vector[i] + sums[i];	//tmpvec
		}
		lt[blockIdx.x] = lt[blockIdx.x] + value;
		//lt[blockIdx.x + gridDim.x * blockIdx.y] = max;
	}

}

/*
 * P = Ltmax + log(sum(exp(Lt-Ltmax)))
 *
 * Should be launched with A blocks of B by nmmodels threads where A * B = nsamples;
 * Shared mem = (nmmodels+1) * blockDim.x * sizeof(real)
 *
 * work should be at least 2 * A * sizeof(real);
 */
__global__ void updatell(
		real *Lt, natural Ltelemsperrow,
		real *v, natural velemsperrow,
		real *work) {
	real ltvalue =  Lt[blockIdx.x * blockDim.x + threadIdx.x + threadIdx.y * Ltelemsperrow];
	real value = 0.0;
	real max = 0.0;
	real * sums = vector + blockDim.x;
	sums[threadIdx.x + threadIdx.y * blockDim.x] = ltvalue;
	__shared__ bool isLastBlockFinished; //If true, then last block has finished.
	__syncthreads();

	/* Get Ltmax of all models*/
	if (threadIdx.y == 0) {
		max = ltvalue;
		for (int i = 1; i < blockDim.y; i++) {
			if (max < sums[threadIdx.x + i *blockDim.x]) {
				max = sums[threadIdx.x + i *blockDim.x];
			}
		}
		vector[threadIdx.x] = max; //Ltmax
	}

	__syncthreads();
	max = vector[threadIdx.x];
	sums[threadIdx.x + threadIdx.y * blockDim.x] = exp(ltvalue - max);

	/* Sum by model */
	__syncthreads();
	if (threadIdx.y == 0) {
		value = 0.0;
		for (int i = 0; i < blockDim.y; i++) {
			value += sums[threadIdx.x + i * blockDim.x];
		}
		vector[threadIdx.x] = max + log(value); // P
		//work[threadIdx.x + blockDim.x * blockIdx.x] = vector[threadIdx.x];
	}

	/* Sum the whole block and reduce it to a single value */
	__syncthreads();
	v[threadIdx.x + blockDim.x * blockIdx.x + threadIdx.y * velemsperrow] = 1/exp(vector[threadIdx.x] - ltvalue);

	if (threadIdx.y == 0 && threadIdx.x == 0) {
		value = 0.0;
		for (int i = 0; i < blockDim.x; i++) {
			value += vector[i];
		}
		work[blockIdx.x] = value;
		natural value = atomicInc(&blocksFinished, gridDim.x);
		isLastBlockFinished = (value == gridDim.x-1);
	}

	__syncthreads();
	if (isLastBlockFinished) {
//		if (threadIdx.y == 0) {
//			value = 0.0;
//			// Reduce till everything fits in threads
//			for (int i = threadIdx.x; i < gridDim.x; i+=blockDim.x) {
//				value += work[i];
//			}
//			sums[threadIdx.x] = value;
//		}
//		__syncthreads();
		if (threadIdx.y == 0 && threadIdx.x == 0) {
			value = 0.0;
			// Reduce till everything fits in threads
			for (int i = 0; i < gridDim.x; i++) {
				value += work[i];
			}
			work[0] = value;
			blocksFinished = 0;
		}

	}
}


/*
 * Multiply each element by itself
 *
 * b = a.*a;
 *
 * Should be launched with N blocks of channels by M threads where M * N = nsamples
 */
__global__ void pow2(real * a, natural aelemsperrow, real * b, natural belemsperrow) {
	real val = a[(blockIdx.x * blockDim.y + threadIdx.y) * aelemsperrow + threadIdx.x];
	b[(blockIdx.x * blockDim.y + threadIdx.y) * belemsperrow + threadIdx.x] = val * val;
}


/*
 * u = v .* z;
 *
 * Should be launched with N by O blocks of channels by M threads where M * N = nsamples and O = nsdm
 * Shared mem = blockDim.y
 *
 */
__global__ void computeu(real * v,
		real * z, natural zelemsperrow,
		real * u, natural uelemsperrow) {
	if (threadIdx.y == 0 && threadIdx.x < blockDim.y) {
		vector[threadIdx.x] = v[blockIdx.x * blockDim.y + threadIdx.x];
	}
	__syncthreads();
	real vvalue = vector[threadIdx.y];
	natural index = (blockIdx.y * (blockDim.y * gridDim.x) + blockDim.y * blockIdx.x + threadIdx.y);

	u[index * uelemsperrow + threadIdx.x] = z[index * zelemsperrow + threadIdx.x] * vvalue;
}


/*
 * usums = sum(u)
 * Performs sum by channel
 *
 * usumwork must be at least an M by channel matrix.
 *
 * Should be launched with N by M blocks of channel threads, where M = nsdm;
 */
__global__ void getusum(real * u, natural uelemsperrow,
		real * usumwork, natural usumworkelemsperrow,
		real * usum, natural usumelemsperrow,
		natural nsamples, natural nsamplesperblock) {
	real sum = 0.0;
	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts
	__shared__ bool isLastBlockFinished; //If true, then last block has finished.
	natural zoffset = blockIdx.y * nsamples * uelemsperrow;

	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
		end = nsamples;
	}
	for (; i < end; i++) {
		sum += u[zoffset + i*uelemsperrow + threadIdx.x];
	}
	usumwork[(blockIdx.x + blockIdx.y * gridDim.x) * usumelemsperrow + threadIdx.x] = sum;

	if (threadIdx.x == 0) {
		natural value = atomicInc(&blocksFinished, gridDim.x);
		isLastBlockFinished = (value == gridDim.x-1);
	}

	__syncthreads();
	if (isLastBlockFinished) {
		for (int m = 0; m < gridDim.y; m++) {
			sum = 0.0;
			for (i = 0; i < gridDim.x; i++) {
				sum += usumwork[threadIdx.x + (i + m * gridDim.x) * usumworkelemsperrow];
			}
			usum[threadIdx.x + (m * usumelemsperrow)] = sum;

		}
		if (threadIdx.x == 0) {
			blocksFinished = 0;
		}
	}
}

/*
 * fp = rho * sign(y) * abs(y) ^ (rho-1);
 * ufp = fp * u;
 *
 * Should be launched with N by O blocks of channels by M threads where M * N = nsamples and O = nsdm
 */
__global__ void computefp(
		real * y, natural yelemsperrow,
		real * u, natural uelemsperrow,
		real * rho, natural rhoelemsperrow,
		real * fp, natural fpelemsperrow,
		real * ufp, natural ufpelemsperrow
		) {
	natural index = (blockIdx.y * (blockDim.y * gridDim.x) + blockDim.y * blockIdx.x + threadIdx.y);
	real yvalue = y[index * yelemsperrow + threadIdx.x];
	real rhovalue = rho[threadIdx.x + blockIdx.y * rhoelemsperrow];
	real value = rhovalue * copysign(pow(fabs(yvalue), rhovalue -1), yvalue);
	fp[index * fpelemsperrow + threadIdx.x]  = value;
	ufp[index * fpelemsperrow + threadIdx.x]  = value * u[index * uelemsperrow + threadIdx.x];
}


/*
 * g = sqrt(beta)* u
 *
 * Should be launched with N  blocks of channels by M threads where M * N = nsamples
 */
__global__ void computeg(
		real * ufp, natural ufpelemsperrow,
		real * beta,
		real * g, natural gelemsperrow
		) {
	natural index =  blockDim.y * blockIdx.x + threadIdx.y;
	if (threadIdx.y == 0) {
		vector[threadIdx.x] = sqrt(beta[threadIdx.x]);
	}
	__syncthreads();
	real ufpvalue = ufp[index * ufpelemsperrow + threadIdx.x];
	real value = vector[threadIdx.x];
	g[index * gelemsperrow + threadIdx.x] += value * ufpvalue;
}

/*
 * dkappa_numer = beta * sum(ufp * fp)
 *
 * knwork must be at least an N by channel matrix.
 *
 * Should be launched with N blocks of channel threads;
 */
__global__ void getdkappanumer(
		real * ufp, natural ufpelemsperrow,
		real * fp, natural fpelemsperrow,
		real * beta,
		real * knwork, natural knworkelemsperrow,
		real * kappa_numer,
		natural nsamples, natural nsamplesperblock) {

	__shared__ bool isLastBlockFinished; //If true, then last block has finished.
	real sum = 0.0;
	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts

	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
		end = nsamples;
	}
	for (; i < end; i++) {
		sum += ufp[i*ufpelemsperrow + threadIdx.x] *
				fp[i*fpelemsperrow + threadIdx.x];
	}
	knwork[blockIdx.x  * knworkelemsperrow + threadIdx.x] = sum;

	if (threadIdx.x == 0) {
		natural value = atomicInc(&blocksFinished, gridDim.x);
		isLastBlockFinished = (value == gridDim.x-1);
	}

	__syncthreads();
	if (isLastBlockFinished) {
		sum = 0.0;
		for (i = 0; i < gridDim.x; i++) {
			sum += knwork[threadIdx.x + i * knworkelemsperrow];
		}
		kappa_numer[threadIdx.x] = beta[threadIdx.x ] * sum;
		if (threadIdx.x == 0) {
			blocksFinished = 0;
		}
	}
}

/*
 * dlambda_numer = sum(u * ((fp * y) -1) ^2)
 *
 * lnwork must be at least an N by channel matrix.
 *
 * Should be launched with N blocks of channel threads;
 */
__global__ void getdlambdanumer(
		real * u, natural uelemsperrow,
		real * fp, natural fpelemsperrow,
		real * y, natural yelemsperrow,
		real * lnwork, natural lnworkelemsperrow,
		real * lambda_numer,
		natural nsamples, natural nsamplesperblock) {

	__shared__ bool isLastBlockFinished; //If true, then last block has finished.
	real sum = 0.0;
	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts

	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
		end = nsamples;
	}
	real value;
	for (; i < end; i++) {
		value = fp[i*fpelemsperrow + threadIdx.x] * y[i*yelemsperrow + threadIdx.x];
		value = value -1.0;
		value = value * value;
		sum += u[i*uelemsperrow + threadIdx.x] * value;
	}
	lnwork[blockIdx.x  * lnworkelemsperrow + threadIdx.x] = sum;

	if (threadIdx.x == 0) {
		natural value = atomicInc(&blocksFinished, gridDim.x);
		isLastBlockFinished = (value == gridDim.x-1);
	}

	__syncthreads();
	if (isLastBlockFinished) {
		sum = 0.0;
		for (i = 0; i < gridDim.x; i++) {
			sum += lnwork[threadIdx.x + i * lnworkelemsperrow];
		}
		lambda_numer[threadIdx.x] = sum;
		if (threadIdx.x == 0) {
			blocksFinished = 0;
		}
	}
}

/*
 * Amica13.m lines 302-317
 *
 * each work matrix must be at least an N by channel matrix.
 *
 * Should be launched with N blocks of channel threads;
 */
__global__ void getbetamu(
		real * u, natural uelemsperrow,
		real * fp, natural fpelemsperrow,
		real * ufp, natural ufpelemsperrow,
		real * y, natural yelemsperrow,
		real * mnwork, natural mnworkelemsperrow,
		real * mdwork, natural mdworkelemsperrow,
		real * bdwork, natural bdworkelemsperrow,
		real * mu_numer,
		real * mu_denom,
		real * beta_denom,
//		real * beta_numer,
//		real * usum,
		real * rho,
		real * beta,
		natural nsamples, natural nsamplesperblock
	) {
	__shared__ bool isLastBlockFinished;
	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts

	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
		end = nsamples;
	}
	real rhovalue = rho[threadIdx.x];

	real mdsum = 0.0;
	real mnsum = 0.0;
	real bdsum = 0.0;
	real yvalue;
	real ufpvalue;

	for (; i < end; i++) {
		ufpvalue = ufp[i*ufpelemsperrow + threadIdx.x];
		yvalue =  y[i*yelemsperrow + threadIdx.x];
		mnsum += ufpvalue;
		if (rhovalue <= 2.0) {
			mdsum += ufpvalue / yvalue;
			bdsum += ufpvalue * yvalue;
		} else {
			mdsum += ufpvalue * fp[i*fpelemsperrow + threadIdx.x];
			bdsum += u[i*uelemsperrow + threadIdx.x] * pow(fabs(yvalue), rhovalue);
		}
	}
	mnwork[blockIdx.x  * mnworkelemsperrow + threadIdx.x] = mnsum;
	mdwork[blockIdx.x  * mdworkelemsperrow + threadIdx.x] = mdsum;
	bdwork[blockIdx.x  * bdworkelemsperrow + threadIdx.x] = bdsum;

	if (threadIdx.x == 0) {
		natural value = atomicInc(&blocksFinished, gridDim.x);
		isLastBlockFinished = (value == gridDim.x-1);
	}

	__syncthreads();
	if (isLastBlockFinished) {
		mdsum = 0.0;
		mnsum = 0.0;
		bdsum = 0.0;
		for (i = 0; i < gridDim.x; i++) {
			mnsum += mnwork[threadIdx.x + i * mnworkelemsperrow];
			mdsum += mdwork[threadIdx.x + i * mdworkelemsperrow];
			bdsum += bdwork[threadIdx.x + i * bdworkelemsperrow];
		}
		mu_numer[threadIdx.x] = mnsum;
//		mu_numer[threadIdx.x] = 1000.0 * blockIdx.x + threadIdx.x;
		mu_denom[threadIdx.x] = sqrt(beta[threadIdx.x]) * mdsum;
		real bdval = bdsum;
//		real bnval = usum[threadIdx.x];
		if (rhovalue > 2.0) {
			bdval = rhovalue*bdval;
//			bnval = 0.0;
		}
		beta_denom[threadIdx.x] = bdval;
//		beta_numer[threadIdx.x] = bnval;
//		beta_denom[threadIdx.x] = bdsum;

//		beta_numer[threadIdx.x] = rhovalue <= 2.0 ? usum[threadIdx.x] : 0.0;
//		beta_numer[threadIdx.x] = 0.0;
//		beta_denom[threadIdx.x] = 0.0;
		if (threadIdx.x == 0) {
			blocksFinished = 0;
		}
	}
}

__global__ void getbetanumer(
		real * beta_numer,
		real * usum,
		real * rho
	) {
	real rhovalue = rho[threadIdx.x];
	real bnval = usum[threadIdx.x];
	if (rhovalue > 2.0) {
		bnval = 0.0;
	}
	beta_numer[threadIdx.x] = bnval;
}

/*
 * drho_numer = sum(u * abs(y)^rho * log(abs(y)^rho))
 *
 * rnwork must be at least an N by channel matrix.
 *
 * Should be launched with N blocks of channel threads;
 */
__global__ void getdrhonumer(
		real * u, natural uelemsperrow,
		real * y, natural yelemsperrow,
		real * rho,
		real * dnwork, natural dnworkelemsperrow,
		real * rho_numer,
		natural nsamples, natural nsamplesperblock
	) {

	__shared__ bool isLastBlockFinished; //If true, then last block has finished.
	real svalue = 0.0;
	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts

	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
		end = nsamples;
	}
	real value;
	real rhovalue = rho[threadIdx.x];
	for (; i < end; i++) {
		value = pow(fabs(y[i*yelemsperrow + threadIdx.x]), rhovalue);
		value = value * log(value);
		svalue += u[i*uelemsperrow + threadIdx.x] * value;
	}
	dnwork[blockIdx.x  * dnworkelemsperrow + threadIdx.x] = svalue;

	if (threadIdx.x == 0) {
		natural value = atomicInc(&blocksFinished, gridDim.x);
		isLastBlockFinished = (value == gridDim.x-1);
	}

	__syncthreads();
	if (isLastBlockFinished) {
		svalue = 0.0;
		for (i = 0; i < gridDim.x; i++) {
			svalue += dnwork[threadIdx.x + i * dnworkelemsperrow];
		}
		rho_numer[threadIdx.x] = svalue;
		if (threadIdx.x == 0) {
			blocksFinished = 0;
		}
	}
}

/*
 * a = a + b
 *
 * Should be launched with N blocks of M by O threads A and B should be M by (N*O)
 */
__global__ void acumulate(
		real * a, natural aelemsperrow,
		real * b, natural belemsperrow) {
	natural index = blockIdx.x + blockDim.x * threadIdx.y;
	a[index * aelemsperrow + threadIdx.x] += b[index * belemsperrow + threadIdx.x];
}


/*
 * amica13.m lines 374-371
 *
 * Should be launched with nmmodels blocks of channels threads.
 *
 */
__global__ void updatekappalambda(
		real * dkappa_numer, natural dknelemsperrow,
		real * dusum, natural dusumelemsperrow,
		real * dlambda_numer, natural dlnelemsperrow,
		real * kappa, natural kappaelemsperrow,
		real * lambda, natural lambdaelemsperrow,
		real * alpha, natural alphaelemsperrow,
		real * mu, natural muelemsperrow) {
	real dk;
	real alphavalue;
	real muvalue;
	real usumvalue = dusum[threadIdx.x + blockIdx.x * dusumelemsperrow];
	dk = dkappa_numer[threadIdx.x + blockIdx.x * dknelemsperrow] / usumvalue;
	alphavalue = alpha[threadIdx.x + blockIdx.x * alphaelemsperrow];
	kappa[threadIdx.x + blockIdx.x * kappaelemsperrow]	+= dk * alphavalue;
	muvalue = mu[threadIdx.x + blockIdx.x * muelemsperrow];
	muvalue = muvalue * muvalue * dk + dlambda_numer[threadIdx.x + blockIdx.x * dlnelemsperrow] / usumvalue;
	lambda[threadIdx.x + blockIdx.x * lambdaelemsperrow] += muvalue * alphavalue;

}


/*
 * a = a ./ b
 *
 * Should be launched with N blocks of M by O threads A and B should be M by (N*O)
 */
__global__ void divide(
		real * a, natural aelemsperrow,
		real * b, natural belemsperrow) {
	natural index = blockIdx.x + blockDim.x * threadIdx.y;
	a[index * aelemsperrow + threadIdx.x] = a[index * aelemsperrow + threadIdx.x] / b[index * belemsperrow + threadIdx.x];
}


/*
 * a = a .* b
 *
 * Should be launched with N blocks of M by O threads A and B should be M by (N*O)
 */
__global__ void multiply(
		real * a, natural aelemsperrow,
		real * b, natural belemsperrow) {
	natural index = blockIdx.x + blockDim.x * threadIdx.y;
	a[index * aelemsperrow + threadIdx.x] = a[index * aelemsperrow + threadIdx.x] * b[index * belemsperrow + threadIdx.x];
}


/*
 * Aproximate psi function (digamma)
 * http://en.wikipedia.org/wiki/Digamma_function
 * http://www.uv.es/~bernardo/1976AppStatist.pdf
 *
 * Shift x + 6 to increase accuracy according to
 *
 * psi(x+1) -1/x = psi(x)
 *
 */
__device__ real psi(real xorig) {
	real x = xorig + 6;
	real result = log(x) - 1/(2*x);
	real tmp1 = x*x; // ^2
	result = result - 1/(12*tmp1);
	real tmp2 = tmp1*tmp1; // ^4
	result = result + 1/(120*tmp2);
	tmp2 = tmp1 * tmp2;	// ^6
	result = result - 1/(252*tmp2);
	tmp2 = tmp1 * tmp2;	// ^8
	result = result + 1/(240*tmp2);
	tmp2 = tmp1 * tmp2;	// ^10
	result = result - 5/(660*tmp2);
	tmp2 = tmp1 * tmp2;	// ^12
	result = result + 691/(32760*tmp2);
	tmp2 = tmp1 * tmp2;	// ^14
	result = result - 1/(12*tmp2);
	tmp2 = tmp1 * tmp2;	// ^16
	result = result + 3617/(510*16*tmp2);
	tmp2 = tmp1 * tmp2;	// ^18
	result = result - 43867/(798*18*tmp2);
	result = result - 1/(xorig +5)- 1/(xorig +4)- 1/(xorig +3)- 1/(xorig +2)- 1/(xorig +1) - 1/(xorig);
	return result;
}

/*
 * Amica13.m (lines 381:386)
 *
 * Should be launched with nmmodels blocks of channels threads.
 *
 */
__global__ void updaterho(
		real * rho, natural rhoelemsperrow,
		real * rhonumer, natural rhonumerelemsperrow,
		real * dusum, natural dusumelemsperrow,
		real rholrate, real rhomin, real rhomax) {
	real usumvalue = dusum[threadIdx.x + blockIdx.x * dusumelemsperrow];
	real rhovalue = rho[blockIdx.x * rhoelemsperrow + threadIdx.x];
	real psivalue = psi(1.0 + 1.0/rhovalue);
	real nrhovalue = 1.0 - (rhovalue/psivalue) * rhonumer[blockIdx.x * rhonumerelemsperrow + threadIdx.x] / usumvalue;
	rhovalue = rhovalue + rholrate * nrhovalue;
	if (rhovalue > rhomax) {
		rhovalue = rhomax;
	} else if (rhovalue < rhomin) {
		rhovalue = rhomin;
	}
	rho[blockIdx.x * rhoelemsperrow + threadIdx.x] = rhovalue;

}

/*
 * c = a .* b
 *
 * Should be launched with N blocks of M by O threads A and B should be M by (N*O)
 */
__global__ void multiplyTo(
		real * a, natural aelemsperrow,
		real * b, natural belemsperrow,
		real * c, natural celemsperrow) {
	natural index = blockIdx.x + blockDim.x * threadIdx.y;
	c[index * celemsperrow + threadIdx.x] = a[index * aelemsperrow + threadIdx.x] * b[index * belemsperrow + threadIdx.x];
}

/*
 * Get bflag: compares each element and verifies if some is <= 1. Adds 1 to bflag if it happens.
 *
 * Should be launched with channels blocks of channels threads.
 *
 */
__global__ void getbflag(real *a, natural aelemsperrow, unsigned int * bflag) {
	if (a[threadIdx.x + blockIdx.x * aelemsperrow] <= 1) {
		atomicAdd(bflag, 1);
	}
}


__global__ void getB(
		real * phi, natural phielemsperrow,
		real * noms, natural nomselemsperrow,
		real * denoms, natural denomselemsperrow,
		real * b, natural belemsperrow
		) {
	real phivalue = -phi[blockIdx.x * phielemsperrow + threadIdx.x];
	real phivaluet = -phi[blockIdx.x + phielemsperrow * threadIdx.x]; //PUAJJJJ (bis)

	b[blockIdx.x * belemsperrow + threadIdx.x] = (noms[threadIdx.x + blockIdx.x * nomselemsperrow] * phivalue + phivaluet)/(denoms[threadIdx.x + blockIdx.x * denomselemsperrow]-1.0);
}

__global__ void updateBDiagonal(
		real * phi, natural phielemsperrow,
		real * lambda,
		real * b, natural belemsperrow
		) {
	 b[threadIdx.x * belemsperrow + threadIdx.x] = (1.0-phi[threadIdx.x * phielemsperrow + threadIdx.x]) / lambda[threadIdx.x];
}


__global__ void geteyemphi(
		real * phi, natural phielemsperrow,
		real * b, natural belemsperrow
		) {
	real phivalue = phi[blockIdx.x * phielemsperrow + threadIdx.x];
	real dvalue = 0;
	if (threadIdx.x == blockIdx.x) {
		dvalue = 1;
	}

	b[blockIdx.x * belemsperrow + threadIdx.x] = dvalue - phivalue;
}


/*
 * Normalize A, beta, mu, by A channel
 *
 * Should be launched with nchannels blocks of nchannels threads;
 * Shared mem = nchannels * sizeof(real)
 */
__global__ void normalizeAsave (
		real * a, natural aelemsperrow,
		real * norms
	) {
	real data = a[blockIdx.x * aelemsperrow + threadIdx.x];
	column[threadIdx.x] = data * data;
	__syncthreads();
	if (threadIdx.x == 0) {
		real sum = column[0];
		for (int i = 1; i < blockDim.x; i++) {
			sum += column[i];
		}
		column[0] = sqrt(sum);
	}
	__syncthreads();
	real norm = column[0];
	a[blockIdx.x * aelemsperrow + threadIdx.x] = data/norm;
	norms[blockIdx.x] = norm;
}


__global__ void normalizemubeta (
		real * norms,
		real * mu, natural muelemsperrow,
		real * beta, natural betaelemsperrow
	) {
	real norm = norms[threadIdx.x];
	mu[blockIdx.x * muelemsperrow + threadIdx.x] *= norm;
	beta[blockIdx.x * betaelemsperrow + threadIdx.x] = beta[blockIdx.x * betaelemsperrow + threadIdx.x] / (norm*norm);
}


/*
 * c = cnew - c
 *
 * Should be launched with nmmodels blocks of nchannels threads
 */
__global__ void cnewminusc (
		real * c, natural celemsperrow,
		real * cnew, natural cnewelemsperrow
	) {
	c[blockIdx.x * celemsperrow + threadIdx.x] = cnew[blockIdx.x * cnewelemsperrow + threadIdx.x] - c[blockIdx.x * celemsperrow + threadIdx.x];
}


/*
 * Substract value to each sample acording to the
 * corresponding element in the vector
 * Should be launched with 1 dimensions blocks. One thread.y for each channel.
 */
__global__ void substractdmu(real *a, natural elemsperrow, real * values) {
	real value = 0.0;
	if (threadIdx.y == 0) {
		value = values[threadIdx.x];
		vector[threadIdx.x] = value;
	}
	__syncthreads();
	a[elemsperrow *  blockIdx.x + threadIdx.x] -= value;
}


//
///*
// * Lt = Lt + sum(Q, channels)
// *
// * Should be launched with 1 dimension per block. One thread.x for each channel.
// * Shared mem = nchannels * sizeof(real)
// */
//__global__ void updateltsingle(real *q, natural qelemsperrow, real * lt, natural ltelemsperrow) {
//	natural sample = blockIdx.x + gridDim.x * blockIdx.y;
//	real value =  q[threadIdx.x + sample * qelemsperrow];
//	vector[threadIdx.x] = value;
//	__syncthreads();
//	if (threadIdx.x == 0) {
//		for (int i = 1; i < blockDim.x; i++) {
//			value += vector[i];
//		}
//		lt[blockIdx.x + gridDim.x * blockIdx.y] = lt[blockIdx.x + gridDim.x * blockIdx.y] + value;
//	}
//}
//
//
///*
// * z = 1 / sum(exp(Q-Qj))
// *
// * j = {1 .. nsdm}
// *
// * Should be launched with 2 dimensions per block. One thread.y for each sdm. One thread.x for each channel.
// * Shared mem = nsdm * nchannels * sizeof(real)
// */
//__global__ void computeu(real *q, natural qelemsperrow, real * u, natural uelemsperrow) {
//	natural sample = blockIdx.x + gridDim.x * blockIdx.y;
//	real value =  q[threadIdx.x + threadIdx.y * qelemsperrow * gridDim.x * gridDim.y + sample * qelemsperrow];
//	real * sums = vector;
//	sums[threadIdx.x + threadIdx.y * blockDim.x] = value;
//	//real * exps = vector + blockDim.y * blockDim.x;
//	__syncthreads();
//	real qj = 0.0;
//	for (int j = 0; j < blockDim.y; j++) {
//		qj += exp(sums[threadIdx.x + blockDim.x * j] - value);
//	}
//	natural sdmoffset = uelemsperrow * (gridDim.x * gridDim.y);
//	natural sampleoffset = uelemsperrow * (gridDim.x * blockIdx.y + blockIdx.x);
//
//	u[threadIdx.x + threadIdx.y * sdmoffset + sampleoffset] = 1/qj;
//}
//

//
///*
// * v = 1 ./ sum(exp(Lts-lth))
// * h = mmodel
// *
// * Should be launched with 1 dimensions per block. One thread.y for each model, one thread.x for each sample.
// * Shared mem = nmmodels * blockDim.x * sizeof(real);
// */
//__global__ void updatev(real * lts, natural ltselemsperrow, real *v, natural velemsperrow) {
//	int sample = blockIdx.x * blockDim.x + threadIdx.x;
//	real value =  lts[sample + threadIdx.y * ltselemsperrow];
//	vector[threadIdx.x + threadIdx.y * blockDim.x] = value;
//	__syncthreads();
//	real sumexp = 0.0;
//	for (int i = 0; i < blockDim.y ; i++) {
//		sumexp += exp(vector[threadIdx.x + i * blockDim.x] - value);
//	}
//	v[sample + threadIdx.y * velemsperrow] = 1/sumexp;
//}
//
///*
// * z = v .* z
// *
// * Should be launched with 3 dimensions per block:
// * 		1 threadIdx.x for each channel
// * 		1 threadIdx.y for each sample
// * 		1 threadIdx.z for each sdm
// * Using two dimension grid;
// *
// * Shared mem = blockDim.y * sizeof(real)
// */
//__global__ void updatezmultimodelsdm(real * z, natural zelemsperrow, real *v) {
//	int sample = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y;
//	real value;
//	if (threadIdx.x == 0 && threadIdx.z == 0) {
//		value =  v[sample];
//		vector[threadIdx.y] = value;
//	}
//	__syncthreads();
//	value = vector[threadIdx.y];
//	//sdm * channels * samples + cursample * channels;
//	int offset = threadIdx.z * zelemsperrow * gridDim.x * gridDim.y * blockDim.y + sample * zelemsperrow;
//	z[offset + threadIdx.x] = z[offset + threadIdx.x] * value;
//}
//
///*
// * z = v
// *
// * Should be launched with 3 dimensions per block:
// * 		1 threadIdx.x for each channel
// * 		1 threadIdx.y for each sample
// * 		1 threadIdx.z for each sdm
// * Using two dimension grid;
// *
// * Shared mem = blockDim.y * sizeof(real)
// */
//__global__ void updatezmultimodel(real * z, natural zelemsperrow, real *v) {
//	int sample = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y;
//	real value;
//	if (threadIdx.x == 0 && threadIdx.z == 0) {
//		value =  v[sample];
//		vector[threadIdx.y] = value;
//	}
//	__syncthreads();
//	value = vector[threadIdx.y];
//	//sdm * channels * samples + cursample * channels;
//	int offset = threadIdx.z * zelemsperrow * gridDim.x * gridDim.y * blockDim.y + sample * zelemsperrow;
//	z[offset + threadIdx.x] = value;
//}
//
//

//
///*
// * sums = sum(z)
// * alpha = sums / value
// *
// * Performs sum by channel by sdm
// *
// * sums must be at least an N * sdm by channel matrix.
// *
// * Should be launched with N blocks of nchannels by nsdm threads;
// */
//__global__ void csumalpha(real * z, natural zelemsperrow, real * sums, natural sumelemsperrow, natural nsamples, natural nsamplesperblock, real * alpha, natural alphaelemsperrow, real value) {
//	real sum = 0.0;
//	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
//	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts
//	natural zoffset = threadIdx.y * nsamples * zelemsperrow;	//sdm * samples * channels;
//	natural sumsoffset = threadIdx.y * gridDim.x * sumelemsperrow; //sdm * blocks * channels;
//	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
//		end = nsamples;
//	}
//	for (; i < end; i++) {
//		sum += z[zoffset + (i*zelemsperrow) + threadIdx.x];
//	}
//	sums[sumsoffset + blockIdx.x * sumelemsperrow + threadIdx.x] = sum;
//
//	if (threadIdx.x == 0 && threadIdx.y == 0) {
//		natural value = atomicInc(&blocksFinished, gridDim.x);
//		isLastBlockFinished = (value == gridDim.x-1);
//	}
//
//	__syncthreads();
//	if (isLastBlockFinished) {
//		sum = 0.0;
//		for (i = 0; i < gridDim.x; i++) {
//			sum += sums[sumsoffset + threadIdx.x + i * sumelemsperrow];
//		}
//		sums[sumsoffset + threadIdx.x] = sum;
//		alpha[threadIdx.x + threadIdx.y * alphaelemsperrow] = sum/value;
//		if (threadIdx.x == 0 && threadIdx.y == 0) {
//			blocksFinished = 0;
//		}
//	}
//}
//
///*
// * z = z / sumz;
// *
// * Should be launched with N blocks of nchannels by nsdm threads;
// *
// * Shared mem = nchannels * sizeof(real) * nsdm
// */
//__global__ void normz(real *z, natural zelemsperrow, natural nsamples, natural nsamplesperblock, real * sumz, natural sumzelemsperrow) {
//	natural i = nsamplesperblock * blockIdx.x;					// Starts when it should
//	natural end = nsamplesperblock * (blockIdx.x + 1);			// Ends when the next starts
//	natural zoffset = threadIdx.y * nsamples * zelemsperrow;	//sdm * samples * channels;
//	vector[threadIdx.x + blockDim.x * threadIdx.y] = sumz[threadIdx.x + threadIdx.y * sumzelemsperrow];
//	__syncthreads();
//	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
//		end = nsamples;
//	}
//	real value = vector[threadIdx.x + blockDim.x * threadIdx.y];
//	if (value > 0) {
//		for (; i < end; i++) {
//			z[zoffset + (i*zelemsperrow) + threadIdx.x] = z[zoffset + (i*zelemsperrow) + threadIdx.x]/value;
//		}
//	}
//}
//
//
///*
// * fp = rho * sign(y) * abs(y) ^ (rho-1);
// * zfp = fp * z;
// *
// * Should be launched with N blocks of nchannels by nsdm threads;
// *
// */
//__global__ void getfp(
//		real * fp,
//		natural fpelemsperrow,
//		real * zfp,
//		natural zfpelemsperrow,
//		real * y,
//		natural yelemsperrow,
//		real * z,
//		natural zelemsperrow,
//		natural nsamples,
//		natural nsamplesperblock,
//		real * rho,
//		natural rhoelemsperrow) {
//	natural i = nsamplesperblock * blockIdx.x;					// Starts when it should
//	natural end = nsamplesperblock * (blockIdx.x + 1);			// Ends when the next starts
//	natural yoffset = threadIdx.y * nsamples * yelemsperrow;	//sdm * samples * channels;
//	natural zoffset = threadIdx.y * nsamples * zelemsperrow;	//sdm * samples * channels;
//	natural fpoffset = threadIdx.y * nsamples * fpelemsperrow;	//sdm * samples * channels;
//	natural zfpoffset = threadIdx.y * nsamples * zfpelemsperrow;//sdm * samples * channels;
//
//	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
//		end = nsamples;
//	}
//	real rhovalue = rho[threadIdx.x + threadIdx.y * rhoelemsperrow];;
//	real yvalue = 0.0;
//	real value = 0.0;
//	for (; i < end; i++) {
//		yvalue = y[yoffset + (i*yelemsperrow) + threadIdx.x];
//		value = rhovalue * copysign(pow(fabs(yvalue), rhovalue -1), yvalue);
//		fp[fpoffset + (i*fpelemsperrow) + threadIdx.x] = value;
//		zfp[zfpoffset + (i*fpelemsperrow) + threadIdx.x] = z[zoffset + (i*yelemsperrow) + threadIdx.x] * value;
//
//	}
//
//}
//
//
///*
// * g = alpha .* sqrt(beta) .* zfp
// *
// * Should be launched with 2 dimensions blocks. One thread.x for each channel.
// */
//__global__ void getg(
//		real * g,
//		natural gelemsperrow,
//		real * zfp,
//		natural zfpelemsperrow,
//		real * alpha,
//		natural alphaelemsperrow,
//		real * beta,
//		natural betaelemsperrow,
//		natural nsdm
//		) {
//	natural sample = threadIdx.y + blockIdx.x * blockDim.y;
//	natural nsamples = gridDim.x * blockDim.y;
//	real * salpha = vector;
//	real * sbeta = &(vector[nsdm * blockDim.x]);
//	if (threadIdx.y < nsdm) { // TODO: NSDM < blockDim.y?
//		salpha[threadIdx.x + blockDim.x * threadIdx.y] = alpha[threadIdx.x + threadIdx.y * alphaelemsperrow];
//		sbeta[threadIdx.x + blockDim.x * threadIdx.y] = sqrt(beta[threadIdx.x + threadIdx.y * betaelemsperrow]);
//	}
//	__syncthreads();
//	real value = 0;
//	for (natural i = 0; i < nsdm; i++) {
//		value = value +
//				salpha[threadIdx.x + i * blockDim.x] *
//				sbeta[threadIdx.x + i * blockDim.x] *
//				zfp[sample * zfpelemsperrow + threadIdx.x + nsamples * zfpelemsperrow * i];
//
//	}
//	g[threadIdx.x + gelemsperrow * sample] = value;
//}
//
//
///*
// * kp = sum(zfp * fp); zfpy2 = sum(z*(fp * y -1)^2)
// *
// * m3dwork1 and m3dwork2 must be at least channel by N by nsdm.
// *
// * Should be launched with N blocks of channel by nsdm threads;
// */
//__global__ void getkpzfpy2(
//		real * z,
//		natural zelemsperrow,
//		real * y,
//		natural yelemsperrow,
//		real * zfp,
//		natural zfpelemsperrow,
//		real * fp,
//		natural fpelemsperrow,
//		real * kp,
//		natural kpelemsperrow,
//		real * zfpy2,
//		natural zfpy2elemsperrow,
//		real * m3dwork1,
//		natural m3dwork1elemsperrow,
//		real * m3dwork2,
//		natural m3dwork2elemsperrow,
//		natural nsamples,
//		natural nsamplesperblock
//	) {
//	real kpsum = 0.0;
//	real zfpy2sum = 0.0;
//	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
//	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts
//	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
//		end = nsamples;
//	}
//	for (; i < end; i++) {
//		kpsum += zfp[(i*zfpelemsperrow) + threadIdx.x + nsamples*zfpelemsperrow*threadIdx.y] *
//				fp[(i*fpelemsperrow) + threadIdx.x + nsamples*fpelemsperrow*threadIdx.y];
//		zfpy2sum += z[(i*zelemsperrow) + threadIdx.x + nsamples*zelemsperrow*threadIdx.y] *
//						pow(fp[(i*fpelemsperrow) + threadIdx.x + nsamples*fpelemsperrow*threadIdx.y] *
//						y[(i*yelemsperrow) + threadIdx.x + nsamples*yelemsperrow*threadIdx.y]
//						  -1, 2);
//	}
//
//
//	m3dwork1[blockIdx.x * m3dwork1elemsperrow + threadIdx.x + gridDim.x*m3dwork1elemsperrow*threadIdx.y] = kpsum;
//	m3dwork2[blockIdx.x * m3dwork2elemsperrow + threadIdx.x + gridDim.x*m3dwork2elemsperrow*threadIdx.y] = zfpy2sum;
//
//	if (threadIdx.x == 0 && threadIdx.y == 0) {
//		natural value = atomicInc(&blocksFinished, gridDim.x);
//		isLastBlockFinished = (value == gridDim.x-1);
//	}
//
//	__syncthreads();
//	if (isLastBlockFinished) {
//		kpsum = 0.0;
//		zfpy2sum = 0.0;
//		for (i = 0; i < gridDim.x; i++) {
//			kpsum += m3dwork1[threadIdx.x + i * m3dwork1elemsperrow + gridDim.x*m3dwork1elemsperrow*threadIdx.y];
//			zfpy2sum += m3dwork2[threadIdx.x + i * m3dwork2elemsperrow + gridDim.x*m3dwork2elemsperrow*threadIdx.y];
//		}
//		kp[threadIdx.x + kpelemsperrow*threadIdx.y] = kpsum;
//		zfpy2[threadIdx.x +  zfpy2elemsperrow*threadIdx.y] = zfpy2sum;
//		if (threadIdx.x == 0 && threadIdx.y == 0) {
//			blocksFinished = 0;
//		}
//	}
//}
//
///*
// * kappa = alpha .* beta .* kp
// * lambda = alpha .* kfpy2 + mu .^2 * beta .* kp
// *
// * Should be launched with 1 block of channel by nsdm threads;
// */
//__global__ void getkappalambda(
//		real * kp,
//		natural kpelemsperrow,
//		real * zfpy2,
//		natural zfpy2elemsperrow,
//		real * alpha,
//		natural alphaelemsperrow,
//		real * beta,
//		natural betaelemsperrow,
//		real * mu,
//		natural muelemsperrow,
//		real * kappa,
//		real * lambda,
//		natural nsdm
//		) {
//	real * salpha = vector;
//	real * sbeta = &(vector[nsdm * blockDim.x]);
//	real * smu2 = &(vector[nsdm * 2 * blockDim.x]);
//	salpha[threadIdx.x + blockDim.x * threadIdx.y] = alpha[threadIdx.x + threadIdx.y * alphaelemsperrow];
//	sbeta[threadIdx.x + blockDim.x * threadIdx.y] = beta[threadIdx.x + threadIdx.y * betaelemsperrow];
//	smu2[threadIdx.x + blockDim.x * threadIdx.y] = mu[threadIdx.x + threadIdx.y * muelemsperrow] * mu[threadIdx.x + threadIdx.y * muelemsperrow];
//
//	__syncthreads();
//	if (threadIdx.y == 0) {
//		real valuel = 0;
//		real valuek = 0;
//		for (natural i = 0; i < nsdm; i++) {
//			valuek = valuek +
//					salpha[threadIdx.x + i * blockDim.x] *
//					sbeta[threadIdx.x + i * blockDim.x] *
//					kp[threadIdx.x + kpelemsperrow * i];
//			valuel = valuel +
//					salpha[threadIdx.x + i * blockDim.x] *
//					zfpy2[threadIdx.x + zfpy2elemsperrow * i] +
//					smu2[threadIdx.x + i * blockDim.x] * sbeta[threadIdx.x + i * blockDim.x] * kp[threadIdx.x + kpelemsperrow * i];
//		}
//		lambda[threadIdx.x] = valuel;
//		kappa[threadIdx.x] = valuek;
//	}
//}
//
//
//
//
///*
// * szfpsy = sum(zfp / y); szfp = sum(zfp)
// *
// * m3dwork1 and m3dwork2 must be at least channel by N by nsdm.
// *
// * Should be launched with N blocks of channel by nsdm threads;
// */
//__global__ void getsumsfordm (
//		real * zfp,
//		natural zfpelemsperrow,
//		real * y,
//		natural yelemsperrow,
//		real * szfpsy,
//		natural szfpsyelemsperrow,
//		real * szfp,
//		natural szfpelemsperrow,
//		real * m3dwork1,
//		natural m3dwork1elemsperrow,
//		real * m3dwork2,
//		natural m3dwork2elemsperrow,
//		natural nsamples,
//		natural nsamplesperblock
//	) {
//	real zfpsysum = 0.0;
//	real zfpsum = 0.0;
//	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
//	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts
//	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
//		end = nsamples;
//	}
//	for (; i < end; i++) {
//		zfpsysum += zfp[(i*zfpelemsperrow) + threadIdx.x + nsamples*zfpelemsperrow*threadIdx.y] /
//				y[(i*yelemsperrow) + threadIdx.x + nsamples*yelemsperrow*threadIdx.y];
//		zfpsum += zfp[(i*zfpelemsperrow) + threadIdx.x + nsamples*zfpelemsperrow*threadIdx.y];
//	}
//
//
//	m3dwork1[blockIdx.x * m3dwork1elemsperrow + threadIdx.x + gridDim.x*m3dwork1elemsperrow*threadIdx.y] = zfpsysum;
//	m3dwork2[blockIdx.x * m3dwork2elemsperrow + threadIdx.x + gridDim.x*m3dwork2elemsperrow*threadIdx.y] = zfpsum;
//
//	if (threadIdx.x == 0 && threadIdx.y == 0) {
//		natural value = atomicInc(&blocksFinished, gridDim.x);
//		isLastBlockFinished = (value == gridDim.x-1);
//	}
//
//	__syncthreads();
//	if (isLastBlockFinished) {
//		zfpsysum = 0.0;
//		zfpsum = 0.0;
//		for (i = 0; i < gridDim.x; i++) {
//			zfpsysum += m3dwork1[threadIdx.x + i * m3dwork1elemsperrow + gridDim.x*m3dwork1elemsperrow*threadIdx.y];
//			zfpsum += m3dwork2[threadIdx.x + i * m3dwork2elemsperrow + gridDim.x*m3dwork2elemsperrow*threadIdx.y];
//		}
//		szfpsy[threadIdx.x + szfpsyelemsperrow*threadIdx.y] = zfpsysum;
//		szfp[threadIdx.x +  szfpelemsperrow*threadIdx.y] = zfpsum;
//		if (threadIdx.x == 0 && threadIdx.y == 0) {
//			blocksFinished = 0;
//		}
//	}
//}
//
///*
// * szfpy = sum(zfp * y); szayr = sum(z * abs(y) ^ rho)
// *
// * m3dwork1 and m3dwork2 must be at least channel by N by nsdm.
// *
// * Should be launched with N blocks of channel by nsdm threads;
// */
//__global__ void getsumsfordb (
//		real * zfp,
//		natural zfpelemsperrow,
//		real * y,
//		natural yelemsperrow,
//		real * z,
//		natural zelemsperrow,
//		real * rho,
//		natural rhoelemsperrow,
//		real * szfpy,
//		natural szfpyelemsperrow,
//		real * szayr,
//		natural szayrelemsperrow,
//		real * m3dwork1,
//		natural m3dwork1elemsperrow,
//		real * m3dwork2,
//		natural m3dwork2elemsperrow,
//		natural nsamples,
//		natural nsamplesperblock
//	) {
//	real zfpysum = 0.0;
//	real zayrsum = 0.0;
//	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
//	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts
//	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
//		end = nsamples;
//	}
//	real prho = rho[threadIdx.x + threadIdx.y * rhoelemsperrow];
//	for (; i < end; i++) {
//		zfpysum += zfp[(i*zfpelemsperrow) + threadIdx.x + nsamples*zfpelemsperrow*threadIdx.y] *
//				y[(i*yelemsperrow) + threadIdx.x + nsamples*yelemsperrow*threadIdx.y];
//		zayrsum += z[(i*zelemsperrow) + threadIdx.x + nsamples*zelemsperrow*threadIdx.y] *
//				pow(abs(y[(i*yelemsperrow) + threadIdx.x + nsamples*yelemsperrow*threadIdx.y]), prho);
//	}
//
//
//	m3dwork1[blockIdx.x * m3dwork1elemsperrow + threadIdx.x + gridDim.x*m3dwork1elemsperrow*threadIdx.y] = zfpysum;
//	m3dwork2[blockIdx.x * m3dwork2elemsperrow + threadIdx.x + gridDim.x*m3dwork2elemsperrow*threadIdx.y] = zayrsum;
//
//	if (threadIdx.x == 0 && threadIdx.y == 0) {
//		natural value = atomicInc(&blocksFinished, gridDim.x);
//		isLastBlockFinished = (value == gridDim.x-1);
//	}
//
//	__syncthreads();
//	if (isLastBlockFinished) {
//		zfpysum = 0.0;
//		zayrsum = 0.0;
//		for (i = 0; i < gridDim.x; i++) {
//			zfpysum += m3dwork1[threadIdx.x + i * m3dwork1elemsperrow + gridDim.x*m3dwork1elemsperrow*threadIdx.y];
//			zayrsum += m3dwork2[threadIdx.x + i * m3dwork2elemsperrow + gridDim.x*m3dwork2elemsperrow*threadIdx.y];
//		}
//		szfpy[threadIdx.x + szfpyelemsperrow*threadIdx.y] = zfpysum;
//		szayr[threadIdx.x +  szayrelemsperrow*threadIdx.y] = zayrsum;
//		if (threadIdx.x == 0 && threadIdx.y == 0) {
//			blocksFinished = 0;
//		}
//	}
//}
//
//
///*
// * if (rho <= 2)
// *   if (nsdm > 1 || nmmodels > 1)
// *     mu = mu + (1/sqrt(beta)) * szfp / szfpsy
// *   beta = beta / szfpy
// * else
// *   if (nsdm > 1 || nmmodels > 1)
// *     mu = mu + sqrt(beta) * szfp / (beta * kp)
// *   beta = beta * (rho * szayr) ^ (-2/rho)
// *
// *
// * Should be launched with 1 block of channel by nsdm threads;
// */
//__global__ void updatemubeta(
//		real * kp,
//		natural kpelemsperrow,
//		real * szfpsy,
//		natural szfpsyelemsperrow,
//		real * szfp,
//		natural szfpelemsperrow,
//		real * szfpy,
//		natural szfpyelemsperrow,
//		real * szfpayr,
//		natural szfpayrelemsperrow,
//		real * beta,
//		natural betaelemsperrow,
//		real * mu,
//		natural muelemsperrow,
//		real * rho,
//		natural rhoelemsperrow,
//		natural nsdm,
//		natural nmmodels
//		) {
//	real sbeta;
//	real srho;
//	srho = rho[threadIdx.x + threadIdx.y * rhoelemsperrow];
//	sbeta = beta[threadIdx.x + threadIdx.y * betaelemsperrow];
//	real db;
//	if (srho <= 2) {
//		if (nsdm > 1 || nmmodels > 1) {
//			real dm = szfpsy[threadIdx.x + threadIdx.y * szfpsyelemsperrow];
//			if (dm > 0) {
//				mu[threadIdx.x + threadIdx.y * muelemsperrow] +=
//						(1.0/(sqrt(sbeta))) * szfp[threadIdx.x + threadIdx.y * szfpelemsperrow] / dm;
//			}
//		}
//		db = szfpy[threadIdx.x + threadIdx.y * szfpyelemsperrow];
//		if (db > 0) {
//			beta[threadIdx.x + threadIdx.y * betaelemsperrow] = sbeta/db;
//		}
//	} else {
//		if (nsdm > 1 || nmmodels > 1) {
//			real bkp = sbeta * kp[threadIdx.x + threadIdx.y * kpelemsperrow];
//
//			if (bkp > 0) {
//				mu[threadIdx.x + threadIdx.y * muelemsperrow] +=
//									sqrt(sbeta) * szfp[threadIdx.x + threadIdx.y * szfpelemsperrow] / bkp;
//			}
//		}
//		db = pow(srho * szfpayr[threadIdx.x + threadIdx.y * szfpayrelemsperrow],
//				-2.0/srho);
//		beta[threadIdx.x + threadIdx.y * betaelemsperrow] = sbeta*db;
//	}
//
//}
//
//
///*
// * dr = sum(z .* log(abs(y)^rho) * abs(y)^rho);
// *
// * Should be launched with N blocks of channel by nsdm threads;
// */
//__global__ void getdr(
//		real * y,
//		natural yelemsperrow,
//		real * z,
//		natural zelemsperrow,
//		real * rho,
//		natural rhoelemsperrow,
//		real * dr,
//		natural drelemsperrow,
//		real * m3dwork1,
//		natural m3dwork1elemsperrow,
//		natural nsamples,
//		natural nsamplesperblock
//	) {
//	real drsum = 0.0;
//	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
//	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts
//	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
//		end = nsamples;
//	}
//	real prho = rho[threadIdx.x + threadIdx.y * rhoelemsperrow];
//	real ytmp;
//	for (; i < end; i++) {
//		ytmp = pow(abs(y[(i*yelemsperrow) + threadIdx.x + nsamples*yelemsperrow*threadIdx.y]), prho);
//		drsum += z[(i*zelemsperrow) + threadIdx.x + nsamples*zelemsperrow*threadIdx.y] *
//				log(ytmp) * ytmp;
//	}
//
//
//	m3dwork1[blockIdx.x * m3dwork1elemsperrow + threadIdx.x + gridDim.x*m3dwork1elemsperrow*threadIdx.y] = drsum;
//
//	if (threadIdx.x == 0 && threadIdx.y == 0) {
//		natural value = atomicInc(&blocksFinished, gridDim.x);
//		isLastBlockFinished = (value == gridDim.x-1);
//	}
//
//	__syncthreads();
//	if () {
//		drsum = 0.0;
//		for (i = 0; i < gridDim.x; i++) {
//			drsum += m3dwork1[threadIdx.x + i * m3dwork1elemsperrow + gridDim.x*m3dwork1elemsperrow*threadIdx.y];
//		}
//		dr[threadIdx.x + drelemsperrow*threadIdx.y] = drsum;
//		if (threadIdx.x == 0 && threadIdx.y == 0) {
//			blocksFinished = 0;
//		}
//	}
//}
//
//

//
///*
// * if (rho > 2)
// *   rho = rho + 0.5 * (psi(1+(1/rho))/rho -dr)
// * else
// *   rho = rho + rholrate * (1 - rho * dr / psi(1+(1/rho)))
// *
// * Should be launched with 1 block of channel by nsdm threads;
// *
// */
//__global__ void updaterho(
//		real * dr,
//		natural drelemsperrow,
//		real * rho,
//		natural rhoelemsperrow,
//		real rholrate,
//		real rhomax,
//		real rhomin
//		) {
//	real srho = rho[threadIdx.x + threadIdx.y * rhoelemsperrow];
//	real sdr = dr[threadIdx.x + threadIdx.y * drelemsperrow];
//	real dr2;
//	if (srho > 2) {
//		dr2 = psi(1+(1/srho)) / srho - sdr;
//		if (isnan(dr2) == 0) {
//			srho = srho + 0.5 * dr2;
//			if (srho > rhomax) {
//				 srho = rhomax;
//			} else if (srho < rhomin) {
//				 srho = rhomin;
//			}
//			rho[threadIdx.x + threadIdx.y * rhoelemsperrow] = srho;
//		}
//	} else {
//		dr2 = 1- srho * sdr / psi(1+(1/srho));
//		if (isnan(dr2) == 0) {
//			srho = srho + rholrate * dr2;
//			if (srho > rhomax) {
//				srho = rhomax;
//			} else if (srho < rhomin) {
//				srho = rhomin;
//			}
//			rho[threadIdx.x + threadIdx.y * rhoelemsperrow] = srho;
//		}
//	}
//}
//

//
//
///*
// * sigma2 = sum(b, 2) / nsamples
// *
// * partialsums must be at least an nblocks by channel matrix.
// *
// * Should be launched with N blocks of nchannels threads;
// */
//__global__ void getsigma2(
//		real * b,
//		natural belemsperrow,
//		real * partialsums,
//		natural partialsumselemsperrow,
//		real * sigma2,
//		natural nsamples,
//		natural nsamplesperblock
//	) {
//	real sum = 0.0;
//	natural i = nsamplesperblock * blockIdx.x;			// Starts when it should
//	natural end = nsamplesperblock * (blockIdx.x + 1);	// Ends when the next starts
//	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
//		end = nsamples;
//	}
//	for (; i < end; i++) {
//		sum += b[i* belemsperrow + threadIdx.x];
//	}
//	partialsums[ blockIdx.x * partialsumselemsperrow + threadIdx.x] = sum;
//
//	if (threadIdx.x == 0) {
//		natural value = atomicInc(&blocksFinished, gridDim.x);
//		isLastBlockFinished = (value == gridDim.x-1);
//	}
//
//	__syncthreads();
//	if (isLastBlockFinished) {
//		sum = 0.0;
//		for (i = 0; i < gridDim.x; i++) {
//			sum += partialsums[threadIdx.x + i * partialsumselemsperrow];
//		}
//		sigma2[threadIdx.x] = sum/nsamples;
//		if (threadIdx.x == 0) {
//			blocksFinished = 0;
//		}
//	}
//}
//
///*
// * Multiply each element by its corresponding element: .* in matlab
// *
// * c = a.*b;
// *
// * Should be launched with N blocks of M threads where a, b and c are N by M matrixes
// */
//__global__ void internalprod(real * a, natural aelemsperrow, real * b, natural belemsperrow, real * c, natural celemsperrow) {
//	c[threadIdx.x + blockIdx.x * celemsperrow] = a[threadIdx.x + blockIdx.x * aelemsperrow] * b[threadIdx.x + blockIdx.x * belemsperrow];
//}
//
///*
// * Get bflag: compares each element and verifies if some is > 1. Adds 1 to bflag if it happens.
// */
//__global__ void getbflag(real *a, natural aelemsperrow, unsigned int * bflag) {
//	if (a[threadIdx.x + blockIdx.x * aelemsperrow] <= 1) {
//		atomicAdd(bflag, 1);
//	}
//}
//
///*
// * Updates dA according to lines 285 to 398 (dA = B)
// *
// * Should be launched with nchannels blocks of nchannels threads;
// */
//__global__ void updatedA(
//		real *dA, natural dAelemsperrow,
//		real *denom,
//		natural denomelemsperrow,
//		real *kappa,
//		real *sigma2,
//		real *lambda
//	) {
//	real l = lambda[threadIdx.x];
//	real k = -kappa[blockIdx.x];
//	real s = sigma2[threadIdx.x];
//	real d = denom[threadIdx.x + blockIdx.x * denomelemsperrow] -1;
//	real da = dA[threadIdx.x + blockIdx.x * dAelemsperrow];
//	real tda = dA[blockIdx.x + threadIdx.x * dAelemsperrow];	//PUAJ!
//	real value = 0.0;
//	value = (k * s * da + tda) / d;
//
//	if (threadIdx.x == blockIdx.x) {
//		value = da/l;
//	}
//	dA[threadIdx.x + blockIdx.x * dAelemsperrow] = value;
//
//}
//
//real hostsum(real * values, natural start, natural end) {
//	real result = 0.0;
//	for (natural i = start; i < end; i++) {
//		result += values[i];
//	}
//	return result;
//}
//
///*
// * Normalize A by channel. Use the norm to scale mu and divide beta by pow2
// *
// * Should be launched with nchannels blocks of nchannels threads;
// * Shared mem = nchannels * sizeof(real)
// */
//__global__ void normalizeall(real * a, natural aelemsperrow, real * mu, natural muelemsperrow, real * beta, natural betaelemsperrow, natural nsdm) {
//	real data = a[blockIdx.x * aelemsperrow + threadIdx.x];
//	column[threadIdx.x] = data * data;
//	__syncthreads();
//	if (threadIdx.x == 0) {
//		real sum = column[0];
//		for (int i = 1; i < blockDim.x; i++) {
//			sum += column[i];
//		}
//		column[0] = sqrt(sum);
//	}
//	__syncthreads();
//	real norm = column[0];
//	a[blockIdx.x * aelemsperrow + threadIdx.x] = data/norm;
//	if (threadIdx.x < nsdm) {
//		mu[threadIdx.x * muelemsperrow + blockIdx.x] *= norm;
//		beta[threadIdx.x * betaelemsperrow + blockIdx.x] /= norm*norm;
//	}
//}
//
///*
// * Subtract each element by its corresponding element: - in matlab
// *
// * c = a-b;
// *
// * Should be launched with N blocks of M threads where a, b and c are N by M matrixes
// */
//__global__ void internalsub(real * a, natural aelemsperrow, real * b, natural belemsperrow, real * c, natural celemsperrow) {
//	c[threadIdx.x + blockIdx.x * celemsperrow] = a[threadIdx.x + blockIdx.x * aelemsperrow] - b[threadIdx.x + blockIdx.x * belemsperrow];
//}
//
///*
// * Add each element by its corresponding element: + in matlab
// *
// * c = a+b;
// *
// * Should be launched with N blocks of M threads where a, b and c are N by M matrixes
// */
//__global__ void internaladd(real * a, natural aelemsperrow, real * b, natural belemsperrow, real * c, natural celemsperrow) {
//	c[threadIdx.x + blockIdx.x * celemsperrow] = a[threadIdx.x + blockIdx.x * aelemsperrow] + b[threadIdx.x + blockIdx.x * belemsperrow];
//}

