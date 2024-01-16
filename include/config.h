/*
 * config.h
 *
 *  Created on: Jul 11, 2012
 *      Author: fraimondo
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <tipos.h>
#include <stdio.h>

#define 	MAX_DEVS 		8
#define 	MAX_CHANNELS 	1024
#define 	MAX_MODELS 		16


/*
 * Constant Values
 */


#define 	LOG2 			0.693147180559945
/*
 * When dividing datasets to multiple devices,
 * this is the maximum number of posible chunks
 */
#define 	MAX_SPLITS		8

/*
 * Prints debugging messages
 * Levels 0 - N
 * 1 - Function calls
 * 2 - Memory information
 * 3 - Function calls inside iterations
 */
//~ #define DEBUG 2



/*
 * Use Peer 2 Peer memory access: not implemented (yet)
 */
//#define USEP2P
#define DEFAULT_COMP_THRESH		0.99
#define DEFAULT_IDENT_INTVL		100
#define	DEFAULT_UPDATE_GM		1
#define DEFAULT_UPDATE_ALPHA	1
#define DEFAULT_UPDATE_A		1
#define DEFAULT_UPDATE_MU		1
#define DEFAULT_UPDATE_BETA		1
#define DEFAULT_UPDATE_RHO		1
#define DEFAULT_DO_REPARM		1
#define DEFAULT_REPARM_MU_C		1
#define DEFAULT_FIX_INIT		1
//#define DEFAULT_FIX_INIT		0

#define	DEFAULT_DO_SPHERE		1
#define DEFAULT_MAXITER 		200
#define DEFAULT_NSDM 			3				//m
#define DEFAULT_NMMODELS 		1				//M

#define DEFAULT_DO_NEWTON		1

#define DEFAULT_LRATE0			0.1
#define DEFAULT_LRATEMAX		1.0
#define DEFAULT_LNATRATE		0.1
#define DEFAULT_NEWT_START_ITER 50
#define DEFAULT_RHO_START_ITER	1
#define DEFAULT_LRATEFACT		0.5
#define DEFAULT_BLOCK_SIZE		5000
#define DEFAULT_MINDLL			1e-8
#define DEFAULT_ITERWIN			1

#define DEFAULT_RHO0			1.5
#define DEFAULT_RHOLRATE		0.1
#define DEFAULT_RHOMIN			1
#define DEFAULT_RHOMAX			2

#define DEFAULT_SHOWLL			1
#define DEFAULT_NUMDEC			0
#define DEFAULT_MAXDEC			3

typedef struct config_t {
	/* GPU specific */
	natural			devs[MAX_DEVS];
	natural			devcount;

	/* AMICA_10.M specific inputs */
	char * 			datafile;			//x
	natural			nchannels;			//n
	natural 		nsamples;			//N
	natural			nmmodels;			//M
	natural 		nsdm;				//m
	natural			maxiter;			//maxiter
	natural			do_sphere;			//do_sphere
	natural			do_newton; 			//do_newton
	char *			ainitfile;			//Ainit
	char *			cinitfile;			//cinit
	char *			kinitfile;			//kinit

	/* AMICA_10.M specific outputs */

	char *			spherefile;
	char *			afile;
	char *			weightsfile;
	char *			khindsfile;
	char *			cfile;
	char *			llfile;
	char *			ltallfile;
	char *			gmfile;
	char *			alphafile;
	char *			mufile;
	char *			betafile;
	char * 			rhofile;


	/* AMICA_10.M internal variables */
	real			comp_thresh;
	natural			ident_intvl;

	natural			update_gm;
	natural			update_alpha;
	natural			update_A;
	natural			update_mu;
	natural			update_beta;
	natural			update_rho;

	natural			do_reparm;
	//natural			reparm_mu_c; // UNUSED
	natural			fix_init;

	real			lrate0;
	real			lratemax;
	real			lnatrate;
	natural			newt_start_iter;
	natural			rho_start_iter;
	real			lratefact;
	natural			block_size;
	//real			mindll;			// UNUSED
	//natural			iterwin;	// UNUSED

	natural			blocks_per_gpu_per_model;

	real 			rho0;
	real			rholrate;
	real			rhomin;
	real			rhomax;

	natural			showLL;

	natural			numdec;
	natural			maxdec;

} config;


#ifdef __cplusplus
extern "C" {
#endif
	error			getConfig(char*);
	void			printConfig(void);
	error			freeConfig(void);
	void			help(void);

	extern config 	currentConfig;
	extern char*	programName;

#ifdef __cplusplus
}
#endif



#endif /* CONFIG_H_ */
