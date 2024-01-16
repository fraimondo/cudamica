#include <config.h>
#include <error.h>
#include <stdlib.h>
#include <stdio.h>
#include <params.h>
#include <tools.h>

config currentConfig;

char* programName;

error getConfig(char * filename) {
	error retorno = SUCCESS;
	error newerr = SUCCESS;
	parsedconfigs * parsedConfigs;

	retorno = parseConfig(filename, &parsedConfigs);

	if (retorno == SUCCESS) {
		/* GPU DEVICES */
		if ((newerr = getNaturalList(parsedConfigs, "devices", (natural*)&currentConfig.devs, &currentConfig.devcount)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.devcount = 1;
				currentConfig.devs[0] = 0;
			} else {
				fprintf(stderr,"ERROR: Invalid devices specification (devices)\n");
				retorno = newerr;
			}
		}

		/* Inputs */
		if ((newerr = getString(parsedConfigs, "datafile", &currentConfig.datafile)) != SUCCESS) {
			fprintf(stderr, "ERROR: Invalid data file (datafile)\n");
			retorno = newerr;
		}

		if ((newerr = getNatural(parsedConfigs, "nchannels", &currentConfig.nchannels)) != SUCCESS) {
			fprintf(stderr,"ERROR: Invalid number of channels (nchannels)\n");
			retorno = newerr;
		}

		if ((newerr = getNatural(parsedConfigs, "nsamples", &currentConfig.nsamples)) != SUCCESS) {
			fprintf(stderr,"ERROR: Invalid number of samples (nsamples)\n");
			retorno = newerr;
		}

		if ((newerr = getNatural(parsedConfigs, "do_sphere", &currentConfig.do_sphere)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.do_sphere = DEFAULT_DO_SPHERE;
			} else {
				fprintf(stderr,"ERROR: Invalid sphering value (do_sphere)\n");
				retorno = newerr;
			}

		}

		if ((newerr = getNatural(parsedConfigs, "maxiter", &currentConfig.maxiter)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.maxiter = DEFAULT_MAXITER;
			} else {
				fprintf(stderr,"ERROR: Invalid number of maximum iterations (maxiter)\n");
				retorno = newerr;
			}
		}

		if ((newerr = getNatural(parsedConfigs, "nsdm", &currentConfig.nsdm)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.nsdm = DEFAULT_NSDM;
			} else {
				fprintf(stderr,"ERROR: Invalid number of source density modles (nsdm)\n");
				retorno = newerr;
			}
		}
		if ((newerr = getNatural(parsedConfigs, "nmmodels", &currentConfig.nmmodels)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.nmmodels = DEFAULT_NMMODELS;
			} else {
				fprintf(stderr,"ERROR: Invalid number of ICA mixture models (nmmodels)\n");
				retorno = newerr;
			}
		}

		if ((newerr = getString(parsedConfigs, "spherefile", &currentConfig.spherefile)) != SUCCESS) {
			fprintf(stderr, "ERROR: Invalid sphere file (spherefile)\n");
			retorno = newerr;
		}

		if ((newerr = getNatural(parsedConfigs, "block_size", &currentConfig.block_size)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.block_size		= DEFAULT_BLOCK_SIZE;
			} else {
				fprintf(stderr,"ERROR: Invalid block size (block_size)\n");
				retorno = newerr;
			}
		}

		if ((newerr = getBool(parsedConfigs, "fix_init", &currentConfig.fix_init)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.fix_init		= DEFAULT_FIX_INIT;
			} else {
				fprintf(stderr,"ERROR: Invalid fix_init value (fix_init)\n");
				retorno = newerr;
			}
		}


#define PARSE_FILE(file) \
		if ((newerr = getString(parsedConfigs, str(file), &currentConfig.file)) != SUCCESS) { \
			if (newerr == ERRORNOPARAM) { \
				currentConfig.file = NULL; \
			} else { \
				fprintf(stderr, "ERROR: Invalid file (" #file "\n"); \
				retorno = newerr; \
			} \
		}



		PARSE_FILE(afile)
		PARSE_FILE(weightsfile)
		PARSE_FILE(khindsfile)
		PARSE_FILE(cfile)
		PARSE_FILE(llfile)
		PARSE_FILE(ltallfile)
		PARSE_FILE(gmfile)
		PARSE_FILE(alphafile)
		PARSE_FILE(mufile)
		PARSE_FILE(betafile)
		PARSE_FILE(rhofile)

		//TODO: Get other input/output files from config
		currentConfig.ainitfile = NULL;
		currentConfig.cinitfile = NULL;
		currentConfig.kinitfile = NULL;


		if ((newerr = getBool(parsedConfigs, "do_newton", &currentConfig.do_newton)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.do_newton = DEFAULT_DO_NEWTON;
			} else {
				fprintf(stderr,"ERROR: Invalid update policy (doNewton)\n");
				retorno = newerr;
			}
		}

		if ((newerr = getNatural(parsedConfigs, "update_rho", &currentConfig.update_rho)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.do_newton = DEFAULT_UPDATE_RHO;
			} else {
				fprintf(stderr,"ERROR: Invalid rho update policy (update_rho)\n");
				retorno = newerr;
			}
		}

		if ((newerr = getNatural(parsedConfigs, "rho_start_iter", &currentConfig.rho_start_iter)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.rho_start_iter = DEFAULT_RHO_START_ITER;
			} else {
				fprintf(stderr,"ERROR: Invalid number of iterations to update rho (rho_start_iter)\n");
				retorno = newerr;
			}
		}

		if ((newerr = getNatural(parsedConfigs, "newt_start_iter", &currentConfig.newt_start_iter)) != SUCCESS) {
			if (newerr == ERRORNOPARAM) {
				currentConfig.newt_start_iter = DEFAULT_NEWT_START_ITER;
			} else {
				fprintf(stderr,"ERROR: Invalid number of iterations to do newton (newt_start_iter)\n");
				retorno = newerr;
			}
		}


		freeParsedConfig(parsedConfigs);
	}

	/*
	 * Other default values not parsed
	 */

	currentConfig.comp_thresh 	= DEFAULT_COMP_THRESH;
	currentConfig.ident_intvl 	= DEFAULT_IDENT_INTVL;
	currentConfig.update_gm		= DEFAULT_UPDATE_GM;
	currentConfig.update_alpha 	= DEFAULT_UPDATE_ALPHA;
	currentConfig.update_A		= DEFAULT_UPDATE_A;
	currentConfig.update_mu		= DEFAULT_UPDATE_MU;
	currentConfig.update_beta	= DEFAULT_UPDATE_BETA;
	currentConfig.do_reparm		= DEFAULT_DO_REPARM;

	currentConfig.lrate0			= DEFAULT_LRATE0;
	currentConfig.lratemax			= DEFAULT_LRATEMAX;
	currentConfig.lnatrate			= DEFAULT_LNATRATE;


	currentConfig.lratefact			= DEFAULT_LRATEFACT;

	//currentConfig.mindll			= DEFAULT_MINDLL;
	//currentConfig.iterwin			= DEFAULT_ITERWIN;


	currentConfig.rho0			= DEFAULT_RHO0;
	currentConfig.rholrate		= DEFAULT_RHOLRATE;
	currentConfig.rhomin		= DEFAULT_RHOMIN;
	currentConfig.rhomax		= DEFAULT_RHOMAX;

	currentConfig.showLL		= DEFAULT_SHOWLL;
	currentConfig.numdec		= DEFAULT_NUMDEC;
	currentConfig.maxdec		= DEFAULT_MAXDEC;

	currentConfig.blocks_per_gpu_per_model	=	-1;

	return retorno;
}

void printConfig(void) {
	PRINT_LINE();
	fprintf(stdout, "Configuration parsed\n");
	PRINT_LINE();
	fprintf(stdout, "devices ");
	int i = 0;
	for (i = 0; i < currentConfig.devcount; i++) {
		fprintf(stdout, "%lu ", currentConfig.devs[i]);
	}
	fprintf(stdout, "(%lu)\n", currentConfig.devcount);
	fprintf(stdout, "datafile %s\n", currentConfig.datafile);
	fprintf(stdout, "nchannels %lu\n", currentConfig.nchannels);
	fprintf(stdout, "nsamples %lu\n", currentConfig.nsamples);
	fprintf(stdout, "do_sphere %lu\n", currentConfig.do_sphere);
	fprintf(stdout, "spherefile %s\n", currentConfig.spherefile);
	fprintf(stdout, "weightsfile %s\n", currentConfig.weightsfile);

	fprintf(stdout, "maxiter %lu\n", currentConfig.maxiter);
	fprintf(stdout, "nsdm %lu\n", currentConfig.nsdm);
	fprintf(stdout, "nmmodels %lu\n", currentConfig.nmmodels);
	fprintf(stdout, "do_newton %d\n", currentConfig.do_newton);
	fprintf(stdout, "fix_init %d\n", currentConfig.fix_init);
	fprintf(stdout, "block_size %d\n", currentConfig.block_size);


	PRINT_LINE();
	PRINT_NEWLINE();
}

error freeConfig(void) {
	free(currentConfig.datafile);
	free(currentConfig.spherefile);
	SAFE_FREE(currentConfig.afile);
	SAFE_FREE(currentConfig.weightsfile);
	SAFE_FREE(currentConfig.khindsfile);
	SAFE_FREE(currentConfig.cfile);
	SAFE_FREE(currentConfig.llfile);
	SAFE_FREE(currentConfig.ltallfile);
	SAFE_FREE(currentConfig.gmfile);
	SAFE_FREE(currentConfig.alphafile);
	SAFE_FREE(currentConfig.mufile);
	SAFE_FREE(currentConfig.betafile);
	SAFE_FREE(currentConfig.rhofile);
	SAFE_FREE(currentConfig.ainitfile);
	SAFE_FREE(currentConfig.cinitfile);
	SAFE_FREE(currentConfig.kinitfile);

	return SUCCESS;
}

void help(void) {
	printf("TODO: help for %s\n", programName);
}
