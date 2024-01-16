#include <stdio.h>
#include <stdlib.h>
#include <error.h>
#include <device.h>
#include <params.h>
#include <amica.h>

/*
 *
 * Reqs:
 * Magma ->
 * 		- gfotran
 * 		- atlas
 */


int main(int argc, char *argv[]) {
#ifdef DEBUG
	fprintf(stdout, "Running CUDAAmica with debug level %d\n", DEBUG);
#endif
	programName = argv[0];
	if ( ! isParam("-f", argv, argc)) {
		printf("\nERROR::Script configuration file is mandatory\n\n\n");
		help();
		return ERRORNOFILE;
	}



	char *filename = getParam("-f", argv, argc);

	C_CHECK_RETURN(getConfig(filename));
	printConfig();

	C_CHECK_RETURN(getDevices());
	C_CHECK_RETURN(selectDevice(currentConfig.devs, currentConfig.devcount));

	recalc_config();

	runamica();

	return SUCCESS;
}
