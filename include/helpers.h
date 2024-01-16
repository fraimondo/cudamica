/*
 * helpers.h
 *
 *  Created on: Jul 24, 2012
 *      Author: fraimondo
 */

#ifndef HELPERS_H_
#define HELPERS_H_

#include <device.h>
#include <error.h>

#ifdef __cplusplus
extern "C" {
#endif


error recalc_config();
error autotune(natural device);
error loadMatrix(real * data, natural rows, natural elems, char* filename, bool isdouble = false);
error loadDevMatrix(cudaPitchedPtr ptr, char* filename, bool isdouble = false);

error writeValue(real value, char * filename);
error writeMatrix(cudaPitchedPtr ptr, char * filename);
error writeDevMatrix(cudaPitchedPtr ptr, char * filename);
error writeDevMatrix3d(cudaPitchedPtr ptr, natural zsize, char * filename);
error loadDevMatrix3d(cudaPitchedPtr ptr, natural index, char* filename);

#ifdef __cplusplus
}
#endif



#endif /* HELPERS_H_ */
