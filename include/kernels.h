/*
 * kernels.h
 *
 *  Created on: Oct 12, 2012
 *      Author: fraimondo
 */

#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <config.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Zero a matrix
 */
__global__ void zeros(real * a, natural elemsperrow);

/*
 * Initializes a vector in a constant value
 * Vector dimensions are (nthreads * nblocks)
 */
__global__ void constant1D(real * a, real val);

/*
 * Initializes a matrix in a constant value
 * Matrix dimensions are (nblocks, nthreads)
 */
__global__ void constant2D(real * a, natural elemsperrow, real val);

/*
 * Initializes a matrix in a constant value
 * Matrix dimensions are (Y= nblocks.x, Z= nthreads.y, X=nthreads.x)
 */
__global__ void constant3D(real * a, natural elemsperrow, real val) ;

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
__global__ void getMean(real* data, natural nchannels, natural nsamples, natural nsamplesperblock, natural drowsize, real* sums, natural sumsrowsize);



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
__global__ void subMean(real* data, natural colwidth, const real* means);

/*
 * Sets a matrix to be the eye matrix.
  *
 * Should be launched with N blocks of rows by M threads of channels
 *
 * data: matrix
 * colwidth: real elements per row size
 */
__global__ void eye(real* data, natural colwidth) ;

/*
 * Scales a matrix by scalar
 *
 * Should be launched with N blocks of rows by M threads of columns
 *
 * data: matrix
 * colwidth: real elements per row size
 * scalar: real value to multiply
 */
__global__ void scale(real* data, natural colwidth, real scalar);


/*
 * Computes variances of data.
 * Should be launched with a single block of channel threads
 *
 * From: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 *
 */
__global__ void getvariance(real * data, natural delemsperrow, natural nsamples, real * sphere, natural selemsperrow, real * means);

/*
 * Add eye to matrix
 * a = matrix(C,R)
 * Should be launched with (a, 1) blocks of (b, 1) threads with C = b and R = a
 *
 */
__global__ void addEye(real * a, natural elemsperrow);


/*
 * Multiply and add constants
 * a = matrix(C,R)
 * Should be launched with (a, 1) blocks of (b, 1) threads with C = b and R = a
 */
__global__ void mpaddConstants(real * a, natural elemsperrow, real mult, real add);


/*
 * Multiply each element in the diagonal
 */
__global__ void getDiagonalMult(real *a, natural elemsperrow, natural rows);

/*
 * Normalize A by channel
 *
 * Should be launched with nchannels blocks of nchannels threads;
 * Shared mem = nchannels * sizeof(real)
 */
__global__ void normalize(real * a, natural elemsperrow);

/*
 * Substract value to each sample acording to the
 * corresponding element in the vector
 * Should be launched with 2 dimensions blocks. One thread.y for each channel.
 */
__global__ void substract(real *a, natural elemsperrow, real * values) ;

/*
 * y = sqrt(beta) * (b - mu)
 * Perform the previous function for each sample in b
 * beta is beta[n] and mu is mu[n] where n is the channel value;
 * Depending on the channel, the sample is multiplied and substracted the corersponding
 * beta and mu value.
 * Should be launched with 2 dimensions blocks. One thread.x for each channel.
 * Should be launched with 2 dimensions grids. One block.y for each sdm
 */
__global__ void betabyxminusmu(
		real *y, natural yelemsperrow,
		real * b, natural belemsperrow,
		real * betas, natural betaelemsperrow,
		real* mus, natural muelemsperrow,
		natural block_size);

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
		natural block_size);


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
		real * z, natural zelemsperrow);


/*
 * P = Ltmax + log(sum(exp(Lt-Ltmax)))
 *
 * Should be launched with A blocks of B by nmmodels threads where A * B = nsamples;
 * Shared mem = (nmmodels+1) * blockDim.x * sizeof(real)
 *
 * work should be at least A * sizeof(real);
 */
__global__ void updatell(
		real *Lt, natural Ltelemsperrow,
		real *v, natural velemsperrow,
		real *work);


/*
 * Multiply each element by itself
 *
 * b = a.*a;
 *
 * Should be launched with N blocks of channels by M threads where M * N = nsamples
 */
__global__ void pow2(real * a, natural aelemsperrow, real * b, natural belemsperrow);


/*
 * u = v .* z;
 *
 * Should be launched with N by O blocks of channels by M threads where M * N = nsamples and O = nsdm
 * Shared mem = blockDim.y
 *
 */
__global__ void computeu(real * v, real * z, natural zelemsperrow, real * u, natural uelemsperrow);



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
		natural nsamples, natural nsamplesperblock);

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
		);

/*
 * g = sqrt(beta)* u
 *
 * Should be launched with N  blocks of channels by M threads where M * N = nsamples
 */
__global__ void computeg(
		real * u, natural uelemsperrow,
		real * beta,
		real * g, natural gelemsperrow
		);


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
		natural nsamples, natural nsamplesperblock);


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
		natural nsamples, natural nsamplesperblock);

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
		natural nsamples, natural nsamplesperblock);


__global__ void getbetanumer(
		real * beta_numer,
		real * usum,
		real * rho
	);


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
		natural nsamples, natural nsamplesperblock);


/*
 * a = a + b
 *
 * Should be launched with N blocks of M by O threads A and B should be M by (N*O)
 */
__global__ void acumulate(
		real * a, natural aelemsperrow,
		real * b, natural belemsperrow);


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
		real * mu, natural muelemsperrow
		);

/*
 * a = a ./ b
 *
 * Should be launched with N blocks of M by O threads A and B should be M by (N*O)
 */
__global__ void divide(
		real * a, natural aelemsperrow,
		real * b, natural belemsperrow);

/*
 * a = a .* b
 *
 * Should be launched with N blocks of M by O threads A and B should be M by (N*O)
 */
__global__ void multiply(
		real * a, natural aelemsperrow,
		real * b, natural belemsperrow);

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
		real rholrate, real rhomin, real rhomax);


/*
 * c = a .* b
 *
 * Should be launched with N blocks of M by O threads A and B should be M by (N*O)
 */
__global__ void multiplyTo(
		real * a, natural aelemsperrow,
		real * b, natural belemsperrow,
		real * c, natural celemsperrow);

/*
 * Get bflag: compares each element and verifies if some is <= 1. Adds 1 to bflag if it happens.
 *
 * Should be launched with channels blocks of channels threads.
 *
 */
__global__ void getbflag(real *a, natural aelemsperrow, unsigned int * bflag);

/*
 *  Computes B acording to Amica13.m 393:407
 *  Should be launched with channels blocks of channels threads.
 */
__global__ void getB(
		real * phi, natural phielemsperrow,
		real * noms, natural nomselemsperrow,
		real * denoms, natural denomselemsperrow,
		real * b, natural belemsperrow
		);

/*
 * Updates B diagonal since its different from the rest
 * Should be launched with 1 block of channels threads.
 */
__global__ void updateBDiagonal(
		real * phi, natural phielemsperrow,
		real * lambda,
		real * b, natural belemsperrow
		);

/*
 *  Computes B acording to Eye(n) - Phi
 *  Should be launched with channels blocks of channels threads.
 */
__global__ void geteyemphi(
		real * phi, natural phielemsperrow,
		real * b, natural belemsperrow
		);

__global__ void normalizeAsave (
		real * a, natural aelemsperrow,
		real * norms
	);

__global__ void normalizemubeta (
		real * norms,
		real * mu, natural muelemsperrow,
		real * beta, natural betaelemsperrow
	);

/*
 * c = cnew - c
 *
 * Should be launched with nmmodels blocks of nchannels threads
 */
__global__ void cnewminusc (
		real * c, natural celemsperrow,
		real * cnew, natural cnewelemsperrow
	);


/*
 * Substract value to each sample acording to the
 * corresponding element in the vector
 * Should be launched with 1 dimensions blocks. One thread.y for each channel.
 */
__global__ void substractdmu(real *a, natural elemsperrow, real * values);

///*
// * Lt = Lt + sum(Q, channels)
// *
// * Should be launched with 1 dimension per block. One thread.x for each channel.
// * Shared mem = nchannels * sizeof(real)
// */
//__global__ void updateltsingle(real *q, natural qelemsperrow, real * lt, natural ltelemsperrow);
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
//__global__ void computeu(real *q, natural qelemsperrow, real * u, natural uelemsperrow);
//
///*
// * P = sum(exp(Lt-Ltmax)))
// *
// * Should be launched with 2 dimensions per block. One thread.y for each model. One thread.x for each sample.
// * Shared mem = (nmmodels+1) * blockDim.x * sizeof(real)
// * P = lt[0];
// */
//__global__ void updatell(real *lt, natural ltelemsperrow, real * partialll) ;
//
///*
// * v = 1 ./ sum(exp(Lts-lth))
// * h = mmodel
// *
// * Should be launched with 1 dimensions per block. One thread.y for each model, one thread.x for each sample.
// * Shared mem = nmmodels * blockDim.x * sizeof(real);
// */
//__global__ void updatev(real * lts, natural ltselemsperrow, real *v, natural velemsperrow);
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
// *
// */
//__global__ void updatezmultimodelsdm(real * z, natural zelemsperrow, real *v);
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
//__global__ void updatezmultimodel(real * z, natural zelemsperrow, real *v);
//
//
///*
// * sums = sum(z)
// * Performs sum by channel
// *
// * sums must be at least an N by channel matrix.
// *
// * Should be launched with N blocks of channel threads;
// */
//__global__ void csum(real * z, natural zelemsperrow, real * sums, natural sumelemsperrow, natural nsamples, natural nsamplesperblock);
//
//
///*
// * sums = sum(z)
// * alpha = sums / value
// *
// * Performs sum by channel
// *
// * sums must be at least an N by channel matrix.
// *
// * Should be launched with N blocks of channel threads;
// */
//__global__ void csumalpha(real * z, natural zelemsperrow, real * sums, natural sumelemsperrow, natural nsamples, natural nsamplesperblock, real * alpha, natural alphaelemsperrow, real value);
//
//
///*
// * z = z / sumz;
// *
// * Should be launched with N blocks of nchannels by nsdm threads;
// *
// * Shared mem = nchannels * sizeof(real)
// */
//__global__ void normz(real *z, natural zelemsperrow, natural nsamples, natural nsamplesperblock, real * sumz, natural sumzelemsperrow);
//
///*
// * fp = rho * sign(y) * abs(y) ^ (rho-1);
// * zfp = fp * z;
// *
// * Should be launched with N blocks of nchannels by nsdm threads;
// *
// */
//__global__ void getfp(real * fp, natural fpelemsperrow, real * zfp, natural zfpelemsperrow, real * y, natural yelemsperrow, real * z, natural zelemsperrow, natural nsamples, natural nsamplesperblock, real * rho, natural rhoelemsperrow);
//
//
///*
// * g = alpha .* sqrt(beta) .* zfp
// *
// * Should be launched with 2 dimensions blocks. One thread.x for each channel.
// *
// * Shared mem = nchannels * sizeof(real) * nsdm * 2
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
//		);
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
//	);
//
///*
// * kappa = alpha .* kp
// * lambda = alpha ,* kfpy2 +
// *
// * Shared mem = nchannels * sizeof(real) * nsdm * 3
// *
// * Should be launched with 1 block of channel by nsdm threads;
// */
//
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
//		);
//
///*
// * szfpsy = sum(zfp / y); szfp = sum(zfp)
// *
// * m3dwork1 and m3dwork2 must be at least channel by N by nsdm.
// *
// * Should be launched with N blocks of channel by nsdm threads;
// */
//__global__ void getsumsfordm (
//	real * zfp,
//	natural zfpelemsperrow,
//	real * y,
//	natural yelemsperrow,
//	real * szfpsy,
//	natural szfpsyelemsperrow,
//	real * szfp,
//	natural szfpelemsperrow,
//	real * m3dwork1,
//	natural m3dwork1elemsperrow,
//	real * m3dwork2,
//	natural m3dwork2elemsperrow,
//	natural nsamples,
//	natural nsamplesperblock
//);
//
///*
// * szfpy = sum(zfp * y); szfpay = sum(zfp * abs(y))
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
//		real * szfpayr,
//		natural szfpayrelemsperrow,
//		real * m3dwork1,
//		natural m3dwork1elemsperrow,
//		real * m3dwork2,
//		natural m3dwork2elemsperrow,
//		natural nsamples,
//		natural nsamplesperblock
//);
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
//	real * kp,
//	natural kpelemsperrow,
//	real * szfpsy,
//	natural szfpsyelemsperrow,
//	real * szfp,
//	natural szfpelemsperrow,
//	real * szfpy,
//	natural szfpyelemsperrow,
//	real * szfpayr,
//	natural szfpayrelemsperrow,
//	real * beta,
//	natural betaelemsperrow,
//	real * mu,
//	natural muelemsperrow,
//	real * rho,
//	natural rhoelemsperrow,
//	natural nsdm,
//	natural nmmodels
//);
//
///*
// * dr = sum(z .* log(abs(y)^rho) * abs(y)^rho);
// *
// * Should be launched with N blocks of channel by nsdm threads;
// */
//__global__ void getdr(
//	real * y,
//	natural yelemsperrow,
//	real * z,
//	natural zelemsperrow,
//	real * rho,
//	natural rhoelemsperrow,
//	real * dr,
//	natural drelemsperrow,
//	real * m3dwork1,
//	natural m3dwork1elemsperrow,
//	natural nsamples,
//	natural nsamplesperblock
//);
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
//	real * dr,
//	natural drelemsperrow,
//	real * rho,
//	natural rhoelemsperrow,
//	real rholrate,
//	real rhomax,
//	real rhomin
//);
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
//	);
//
//
///*
// * Multiply each element by its corresponding element: .* in matlab
// *
// * c = a.*b;
// *
// * Should be launched with N blocks of M threads where a, b and c are N by M matrixes
// */
//__global__ void internalprod(real * a, natural aelemsperrow, real * b, natural belemsperrow, real * c, natural belemsperrow);
//
//
///*
// * Get bflag: compares each element and verifies if some is > 1. Adds 1 to bflag if it happens.
// */
//__global__ void getbflag(real *a, natural aelemsperrow, unsigned int * bflag);
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
//	);
//
///*
// * Normalize A by channel. Use the norm to scale mu and divide beta by pow2
// *
// * Should be launched with nchannels blocks of nchannels threads;
// * Shared mem = nchannels * sizeof(real)
// */
//
//__global__ void normalizeall(real * a, natural aelemsperrow, real * mu, natural muelemsperrow, real * beta, natural betaelemsperrow, natural nsdm);
//
//
///*
// * Subtract each element by its corresponding element: - in matlab
// *
// * c = a-b;
// *
// * Should be launched with N blocks of M threads where a, b and c are N by M matrixes
// */
//__global__ void internalsub(real * a, natural aelemsperrow, real * b, natural belemsperrow, real * c, natural celemsperrow);
//
//
///*
// * Add each element by its corresponding element: + in matlab
// *
// * c = a+b;
// *
// * Should be launched with N blocks of M threads where a, b and c are N by M matrixes
// */
//__global__ void internaladd(real * a, natural aelemsperrow, real * b, natural belemsperrow, real * c, natural celemsperrow);
//
///* TODO: Use magma or blas or MKL! */
///*
// * Sums values from values starting from start till end [start, end)
// */
//real hostsum(real * values, natural start, natural end);

#ifdef __cplusplus
}
#endif

#endif /* __KERNELS_H__ */
