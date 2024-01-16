#ifndef __PARAMS_H__
#define __PARAMS_H__
#include <error.h>
#include <config.h>

typedef struct parsedconfigs_t {
	char ** configs;
	int lines;
} parsedconfigs;


#ifdef __cplusplus
extern "C" {
#endif


error parseConfig(char* filename, parsedconfigs** parsed_configs);
error freeParsedConfig(parsedconfigs* parsed_configs);
error getReal(parsedconfigs* config, char* string, real* result);
error getBool(parsedconfigs* config, char* string, natural* result);
error getInt(parsedconfigs* config, char* string, int* result);
error getLong(parsedconfigs* config, char* string, long* result);
error getNatural(parsedconfigs* config, char* string, natural* result);
error getNaturalList(parsedconfigs* config, char* string, natural* result, natural * size);
error getIntList(parsedconfigs* config, char* string, int* result, int * size);
error getLongList(parsedconfigs* config, char* string, long* result, long * size);
error getString(parsedconfigs* config, char* string, char** result);
char* getParam(char * needle, char* haystack[], int count);
int isParam(char * needle, char* haystack[], int count);

#ifdef __cplusplus
}
#endif

#endif
