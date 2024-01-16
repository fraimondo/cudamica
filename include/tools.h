/*
 * tools.h
 *
 *  Created on: Jul 23, 2012
 *      Author: fraimondo
 */

#ifndef TOOLS_H_
#define TOOLS_H_

#include <sys/time.h>

#define PRINT_LINE() \
	fprintf(stdout, "====================================\n")

#define PRINT_NEWLINE() \
	fprintf(stdout, "\n");

#define xstr(x) #x
#define str(x) xstr(x)

#define STRINGIZE(arg)  STRINGIZE1(arg)
#define STRINGIZE1(arg) STRINGIZE2(arg)
#define STRINGIZE2(arg) #arg

#define CONCATENATE(arg1, arg2)   CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2)  CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2)  arg1##arg2

#define FOR_EACH_1(what, x, ...) what(x)
#define FOR_EACH_2(what, x, ...)\
  what(x);\
  FOR_EACH_1(what,  __VA_ARGS__);
#define FOR_EACH_3(what, x, ...)\
  what(x);\
  FOR_EACH_2(what, __VA_ARGS__);
#define FOR_EACH_4(what, x, ...)\
  what(x);\
  FOR_EACH_3(what,  __VA_ARGS__);
#define FOR_EACH_5(what, x, ...)\
  what(x);\
 FOR_EACH_4(what,  __VA_ARGS__);
#define FOR_EACH_6(what, x, ...)\
  what(x);\
  FOR_EACH_5(what,  __VA_ARGS__);
#define FOR_EACH_7(what, x, ...)\
  what(x);\
  FOR_EACH_6(what,  __VA_ARGS__);
#define FOR_EACH_8(what, x, ...)\
  what(x);\
  FOR_EACH_7(what,  __VA_ARGS__);
#define FOR_EACH_9(what, x, ...)\
  what(x);\
  FOR_EACH_8(what,  __VA_ARGS__);
#define FOR_EACH_10(what, x, ...)\
  what(x);\
  FOR_EACH_9(what,  __VA_ARGS__);
#define FOR_EACH_11(what, x, ...)\
  what(x);\
  FOR_EACH_10(what,  __VA_ARGS__);
#define FOR_EACH_12(what, x, ...)\
  what(x);\
  FOR_EACH_11(what,  __VA_ARGS__);
#define FOR_EACH_13(what, x, ...)\
  what(x);\
  FOR_EACH_12(what,  __VA_ARGS__);
#define FOR_EACH_14(what, x, ...)\
  what(x);\
  FOR_EACH_13(what,  __VA_ARGS__);
#define FOR_EACH_15(what, x, ...)\
  what(x);\
  FOR_EACH_14(what,  __VA_ARGS__);
#define FOR_EACH_16(what, x, ...)\
  what(x);\
  FOR_EACH_15(what,  __VA_ARGS__);
#define FOR_EACH_17(what, x, ...)\
  what(x);\
  FOR_EACH_16(what,  __VA_ARGS__);
#define FOR_EACH_18(what, x, ...)\
  what(x);\
  FOR_EACH_17(what,  __VA_ARGS__);
#define FOR_EACH_19(what, x, ...)\
  what(x);\
  FOR_EACH_18(what,  __VA_ARGS__);
#define FOR_EACH_20(what, x, ...)\
  what(x);\
  FOR_EACH_19(what,  __VA_ARGS__);
#define FOR_EACH_21(what, x, ...)\
  what(x);\
  FOR_EACH_20(what,  __VA_ARGS__);
#define FOR_EACH_22(what, x, ...)\
  what(x);\
  FOR_EACH_21(what,  __VA_ARGS__);
#define FOR_EACH_23(what, x, ...)\
  what(x);\
  FOR_EACH_22(what,  __VA_ARGS__);
#define FOR_EACH_24(what, x, ...)\
  what(x);\
  FOR_EACH_23(what,  __VA_ARGS__);

#define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, N, ...) N
#define FOR_EACH_RSEQ_N() 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define FOR_EACH_(N, what, x, ...) CONCATENATE(FOR_EACH_, N)(what, x, __VA_ARGS__)
#define FOR_EACH(what, x, ...) FOR_EACH_(FOR_EACH_NARG(x, __VA_ARGS__), what, x, __VA_ARGS__)

#define SAFE_CUDA_FREE(x) \
	if (x.ptr != NULL) {CUDA_CHECK_RETURN(cudaFree(x.ptr))}

#define SAFE_FREE_HOST(x) \
	if (x.ptr != NULL) {CUDA_CHECK_RETURN(cudaFreeHost(x.ptr))}


//fprintf(stdout, "%s = (%f) | ", str(param), param); \

#define PRINT_KERNEL_PARAMETER(param) \
	fprintf(stdout, "%s = %p | ", str(param), (void*)(param));

//#define PRINT_KERNEL_PARAMETER(param)  fprintf(stdout, "%s = %d ", str(param), 0);


#define OFFSET4D2D(pointer, index, zindex, zsize) \
	(real*)((unsigned long)pointer.ptr + (pointer.pitch * pointer.ysize * zindex * zsize) \
			 + (pointer.pitch * pointer.ysize * index))

#define OFFSET4D(pointer, index, zsize) \
	(real*)((unsigned long)pointer.ptr + (pointer.pitch * pointer.ysize * index * zsize))

#define OFFSET3D(pointer, index) \
	(real*)((unsigned long)pointer.ptr + (pointer.pitch * pointer.ysize * index))

#define OFFSET3D1D(pointer, yindex, zindex) \
	(real*)((unsigned long)pointer.ptr + (pointer.pitch * pointer.ysize * zindex) + (pointer.pitch * (yindex)))

#define OFFSET2D(pointer, index) \
	(real*)((unsigned long)pointer.ptr + (pointer.pitch * (index)))

#define OFFSET1D(pointer, index) \
	(real*)((unsigned long)pointer.ptr + (index))

#define HOFFSET3D(pointer, pitch, ysize, index) \
	(real*)((unsigned long)pointer + (pitch * ysize * index))

#define HOFFSET2D(pointer, pitch, index) \
	(real*)((unsigned long)pointer + (pitch * index))

#define TICKTIMER_START(name) \
		clock_t name##_cstart; \
		clock_t name##_cend; \
		clock_t name##_ticks; \
		name##_cstart = clock();

#define TICKTIMER_END(name) \
		name##_cend = clock(); \
		name##_ticks = name##_cend - name##_cstart; \
		DPRINTF(1, "Elapsed ticks in %s = %lu\n", str(name), name##_ticks);

#define SECTIMER_START(name) \
		struct timeval name##_tstart; \
		struct timeval name##_tend; \
		time_t name##_msecs; \
		gettimeofday(&name##_tstart, NULL);

#define SECTIMER_END(name) \
		gettimeofday(&name##_tend, NULL); \
		{ time_t msecs = (((name##_tend.tv_usec < name##_tstart.tv_usec) ? 1000000 : 0 ) + name##_tend.tv_usec - name##_tstart.tv_usec)/1000;\
		msecs += (((name##_tend.tv_usec < name##_tstart.tv_usec) ? -1 : 0 ) +  name##_tend.tv_sec - name##_tstart.tv_sec) * 1000; \
		name##_msecs = msecs; }; \
		DPRINTF(1, "Elapsed ms in %s = %lu\n", str(name), name##_msecs);

#define DEVICE_SYNC \
	for (natural curdevice = 0; curdevice < gpuCount; curdevice++) {	\
		DPRINTF(1, "Waiting for device %lu to sync... ", curdevice);		\
		CUDA_CHECK_RETURN(cudaSetDevice(gpus[curdevice].realdevice));	\
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());						\
		DPRINTF(1, "Done!\n")											\
	}

#define SAFE_FREE(x) \
	if (x != NULL) {free(x);}

#ifdef DEBUG
	#define PREFIX "DEBUG::%s:%d "
	#define DPRINTF(N, ...) \
		if (N <= DEBUG) { \
			fprintf(stdout, PREFIX ,__FILE__, __LINE__); \
			fprintf(stdout, __VA_ARGS__); \
		}

	#define DBUILD(N, X) \
		if (N <= DEBUG) { \
			X \
		}





#if DEBUG > 2
	#warning "#####################################################"
	#warning "Writing many matrix to files"
	#warning "#####################################################"
	#define DDEVWRITEMAT(stream, ...)		\
		CUDA_CHECK_RETURN(cudaStreamSynchronize(stream)); \
		C_CHECK_RETURN(writeDevMatrix(__VA_ARGS__))

	#define DWRITEMAT(...)		\
		C_CHECK_RETURN(writeMatrix(__VA_ARGS__))

	#define DPRINT_ALLOC(x) \
			fprintf(stdout, PREFIX ,__FILE__, __LINE__); \
			fprintf(stdout, "Malloc %s at %p (%lu by %lu - pitch %lu)\n", str(x), x.ptr, x.xsize, x.ysize, x.pitch ); \

	#define DDEVWRITEMAT3D(stream, ...)		\
		CUDA_CHECK_RETURN(cudaStreamSynchronize(stream)); 	\
		C_CHECK_RETURN(writeDevMatrix3d(__VA_ARGS__));
#else
	#define DDEVWRITEMAT(stream, ...)
	#define DWRITEMAT(...)
	#define DPRINT_ALLOC(x)
	#define DDEVWRITEMAT3D(stream, offset, mat, ...)
#endif


	#define PRINT_NOSTD_KERNEL_CALL(function, blocks, threads, mem, stream, ...) \
			DPRINTF(1, "%s<<<%lu, %lu, %lu, %p>>> -> ", str(function), blocks, threads, mem, stream);\
			FOR_EACH(PRINT_KERNEL_PARAMETER, __VA_ARGS__) \
			fprintf(stdout, "\n");

	#define PRINT_KERNEL_CALL(function, blocks, threads, mem, stream, ...) \
			DPRINTF(1, "%s<<<(%d, %d), (%d, %d, %d), %d, %p>>> -> ", str(function), blocks.x, blocks.y, threads.x, threads.y, threads.z, mem, stream);\
			FOR_EACH(PRINT_KERNEL_PARAMETER, __VA_ARGS__) \
			fprintf(stdout, "\n");

#else
	#define DPRINTF(N, ...)
	#define DBUILD(N, ...)
	#define DPRINT_ALLOC(x)
	#define DWRITEMAT(...)
	#define DDEVWRITEMAT(...)
	#define DDEVWRITEMAT3D(...)
	#define PRINT_NOSTD_KERNEL_CALL(function, blocks, threads, mem, stream, ...)
	#define PRINT_KERNEL_CALL(function, blocks, threads, mem, stream, ...)
#endif

#define DDEVLOADMAT(...)		\
	C_CHECK_RETURN(loadDevMatrix(__VA_ARGS__))

#define DLOADMAT(...)		\
	C_CHECK_RETURN(loadDevMatrix(__VA_ARGS__))

#define DLOADMAT3D(...)		\
	C_CHECK_RETURN(loadDevMatrix3d(__VA_ARGS__))


#endif /* TOOLS_H_ */
