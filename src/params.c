#include <params.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <tools.h>

error getReal(parsedconfigs* config, char* string, real* result) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				*result = (real)atof(item);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}


error getBool(parsedconfigs* config, char* string, natural* result) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				if (strcmp(item, "on") == 0) {
					*result = 1;
				} else if (strcmp(item, "off") == 0) {
					*result = 0;
				} else if (strcmp(item, "on\n") == 0) {
					*result = 1;
				} else if (strcmp(item, "off\n") == 0) {
					*result = 0;
				} else if (strcmp(item, "none") == 0) {
					*result = 2;
				} else if (strcmp(item, "none\n") == 0) {
					*result = 2;
				} else {
					return ERRORINVALIDPARAM;
				}
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}


error getNatural(parsedconfigs* config, char* string, natural* result) {
#ifdef USELONG
	return getLong(config, string, result);
#else
	return getInt(config, string, result);
#endif
}

error getNaturalList(parsedconfigs* config, char* string, natural result[], natural* size) {
#ifdef USELONG
	return getLongList(config, string, result, size);
#else
	return getIntList(config, string, result, size);
#endif
}

error getInteger(parsedconfigs* config, char* string, integer* result) {
#ifdef USELONG
	return getLong(config, string, result);
#else
	return getInt(config, string, result);
#endif
}


error getInt(parsedconfigs* config, char* string, int* result) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				*result = atoi(item);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}

error getIntList(parsedconfigs* config, char* string, int result[], int* size) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				int lcount = 0;
				while ((item = strtok(NULL, " ")) != NULL) {
					result[lcount] = atoi(item);
					lcount++;
				}
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}

error getLongList(parsedconfigs* config, char* string, long result[], long * size) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				int lcount = 0;
				while ((item = strtok(NULL, " ")) != NULL) {
					result[lcount] = atol(item);
					lcount++;
				}
				*size = lcount;
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}


error getLong(parsedconfigs* config, char* string, long* result) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				*result = atol(item);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}


error getString(parsedconfigs* config, char* string, char** result) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (buffer[i] != NULL && (strstr(buffer[i], string) != NULL)) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				int len = strlen(item);
				*result = malloc(len +1);
				memset(*result, 0, len+1);
				strncpy(*result, item, len-1);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}


char* getParam(char * needle, char* haystack[], int count) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strcmp(needle, haystack[i]) == 0) {
			if (i < count -1) {
				return haystack[i+1];
			}	
		}
	}
	return 0;	
}


int isParam(char * needle, char* haystack[], int count) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strcmp(needle, haystack[i]) == 0) {
			return 1;
		}
	}
	return 0;	
}


error parseConfig(char* filename, parsedconfigs** parsed_configs) {
	PRINT_LINE();
	fprintf(stdout, " Opening config file %s\n", filename);
	PRINT_LINE();
	FILE* cfile = fopen(filename, "r");
	error retorno = SUCCESS;
	if (cfile == NULL) {
		retorno = ERRORINVALIDCONFIG;
		*parsed_configs = NULL;
	} else {
		parsedconfigs * result = malloc(sizeof(parsedconfigs));
		char * buffer = malloc(5000);
		int lines = 0;
		while (fgets(buffer, 5000, cfile) != NULL) lines++;
		rewind(cfile);

		char **configs = (char**)malloc(lines * sizeof(char*));
		char *current;
		int i = 0;
		for (i = 0; i < lines; i++) {
			configs[i] = NULL;
			current = fgets(buffer, 5000, cfile);
			if (current != NULL) {
				if (current[0] != '#') {
					configs[i] = (char*)malloc((strlen(current)+1) * sizeof(char));
					strcpy(configs[i], current);
				}
			} else {
				configs[i] = NULL;
			}
		}

		result->configs = configs;
		result->lines = lines;
		free(buffer);
		*parsed_configs = result;
		fclose(cfile);
		fprintf(stdout, "Done!\n");
		retorno = SUCCESS;
		PRINT_LINE();
		PRINT_NEWLINE();
	}

	return retorno;
}

error freeParsedConfig(parsedconfigs* config) {
	int i;
	for (i = 0; i < config->lines; i++) {
		if (config->configs[i] != NULL) {
			free(config->configs[i]);
		}
	}
	
	free(config->configs);
	free(config);
	return SUCCESS;
}
