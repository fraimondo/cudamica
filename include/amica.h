/*
 * amica.h
 *
 *  Created on: Jul 23, 2012
 *      Author: fraimondo
 */

#include <error.h>
#include <helpers.h>
#include <device.h>




#ifndef AMICA_H_
#define AMICA_H_


#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
	/*
	 * Each block of the model
	 */
	natural			device;
	cudaStream_t*	stream;			//stream
	natural			models;			//model number

	natural			start;
	natural			end;
	natural			block_size;

	/* Numerators and denominators */
//	cudaPitchedPtr	dev_dalpha_numer;		//real: nchannels by nsdm by nmmodels */ usumsum
//	cudaPitchedPtr	dev_dalpha_denom;		//real: nchannels by nsdm by nmmodels */ vsumsum
	cudaPitchedPtr	dev_dmu_numer;			//real: nchannels by nsdm by nmmodels */
	cudaPitchedPtr	dev_dmu_denom;			//real: nchannels by nsdm by nmmodels */
	cudaPitchedPtr	dev_dbeta_numer;		//real: nchannels by nsdm by nmmodels */
	cudaPitchedPtr	dev_dbeta_denom;		//real: nchannels by nsdm by nmmodels */
	cudaPitchedPtr	dev_drho_numer;			//real: nchannels by nsdm by nmmodels */
//	cudaPitchedPtr	dev_drho_denom;			//real: nchannels by nsdm by nmmodels */ usumsum

	real			ll;

	cudaPitchedPtr	dev_dlambda_numer;		//real: nchannels by nsdm by nmmodels */ CHECKED
//	cudaPitchedPtr	dev_dlambda_denom;		//real: nchannels by nsdm by nmmodels */ usumsum
	cudaPitchedPtr	dev_dsigma2_numer;		//real: nchannels by nmmodels */ CHECKED
//	cudaPitchedPtr	dev_dsigma2_denom;		//real: nchannels by nmmodels */ vsumsum

	cudaPitchedPtr	dev_dkappa_numer;		//real: nchannels by nsdm by nmmodels */
//	cudaPitchedPtr	dev_dkappa_denom;		//real: nchannels by nsdm by nmmodels */ usumsum

	cudaPitchedPtr	dev_phi;				//real: nchannels by nchannels by nmmodels */
	cudaPitchedPtr	dev_cnew;				//real: nchannels by nmmodels */

	cudaPitchedPtr	dev_v;					//real: blocksize by nmmodels */
	cudaPitchedPtr  dev_usum;				//real: nchannels by nsdm by nmmodel*/
	cudaPitchedPtr 	host_usum;				//real: nchannels by nsdm by nmmodel*/

	cudaPitchedPtr  dev_vsum;				//real: nmmodels CHECKED
	cudaPitchedPtr 	host_vsum;				//real: nmmodels
} model_block_t;

typedef struct {

	natural				master_device;			//device that run this model


	/* Input / Output matrixes (kept between iterations)*/
	cudaPitchedPtr		dev_a;					//real: nchannels by nchannels
	cudaPitchedPtr		dev_c;					//real: nchannels
	cudaPitchedPtr		dev_beta;				//real: nchannels by nsdm
	cudaPitchedPtr		dev_mu;					//real: nchannels by nsdm
	cudaPitchedPtr		dev_alpha;				//real: nchannels by nsdm
	cudaPitchedPtr		dev_rho;				//real: nchannels by nsdm

	cudaPitchedPtr		dev_ltall;				//real: nsamples

	real				gm;

	real				ldet;

	real				host_vsumsum;
	real				host_usumsum;


	/* Intermediate matrixes (overwritten in each iteration) */

	/* Redefine some unused vars to save memory */


	/*
	 * Host pseudoinverse matrix
	 */
	cudaPitchedPtr 		host_pinva;				//real: nchannels x nchannels
	cudaPitchedPtr 		host_pinvu;				//real: nchannels x nchannels
	cudaPitchedPtr 		host_pinvs;				//real: nchannels x 1
	cudaPitchedPtr 		host_pinvsdiag; 		//real: nchannels x nchannels
	cudaPitchedPtr 		host_pinvv;				//real: nchannels x nchannels
	cudaPitchedPtr 		host_pinvsuperb;		//real: nchannels x 1


	/*
	 * Working variables
	 */
	int					info;
	cudaPitchedPtr		host_ipiv;					// int: nchannels x 1

} model_t ;


/*
 * Blocks
 */

extern model_block_t 		*blocks;				//blocks
extern natural				nblocks;				//number of blocks


error runamica(void);


#ifdef __cplusplus
}
#endif

#endif /* AMICA_H_ */
