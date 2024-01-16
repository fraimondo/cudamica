/*
 * tipos.h
 *
 *  Created on: Jul 11, 2012
 *      Author: fraimondo
 */

#ifndef TIPOS_H_
#define TIPOS_H_

#ifdef USELONG
typedef unsigned long natural;
typedef long	integer;
#else
typedef unsigned int natural;
typedef int integer;
#endif

typedef int error;
typedef int boolean;

#ifdef USESINGLE
	typedef float real;
#else
	typedef double real;
#endif

#endif /* TIPOS_H_ */
