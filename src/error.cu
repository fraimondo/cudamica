#include <error.h>
#include <magma.h>


char * errors[] = {
		"Success!",
		"No device memory",
		"No parameter",
		"Invalid parameter",
		"Invalid config",
		"No file"
};

const char * cGetErrorString(natural value) {
	return errors[-value];
}

const char * mklGetErrorString(int value) {
	switch (value) {
		case -1010:
			return "LAPACK_WORK_MEMORY_ERROR";
		case -1011:
			return "LAPACK_TRANSPOSE_MEMORY_ERROR";
		default:
			return "Unknown MKL error";
	}
}

const char * curandGetErrorString(curandStatus_t value) {

	switch (value) {
		case CURAND_STATUS_SUCCESS:
			return "No error";
		case CURAND_STATUS_VERSION_MISMATCH:
			return "Header file and linked library version do not match.";
		case CURAND_STATUS_NOT_INITIALIZED:
			return "Generator not initialized.";
		case CURAND_STATUS_ALLOCATION_FAILED:
			return "Memory allocation failed.";
		case CURAND_STATUS_TYPE_ERROR:
			return "Generator is wrong type.";
		case CURAND_STATUS_OUT_OF_RANGE:
			return "Argument out of range.";
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
			return "Length requested is not a multple of dimension.";
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			return "GPU does not have double precision required by MRG32k3a.";
		case CURAND_STATUS_LAUNCH_FAILURE:
			return "Kernel launch failure.";
		case CURAND_STATUS_PREEXISTING_FAILURE:
			return "Preexisting failure on library entry.";
		case CURAND_STATUS_INITIALIZATION_FAILED:
			return "Initialization of CUDA failed.";
		case CURAND_STATUS_ARCH_MISMATCH:
			return "rchitecture mismatch, GPU does not support requested feature.";
		case CURAND_STATUS_INTERNAL_ERROR:
			return "Internal library error.";
		default:
			return "Unknown error";

	}

}

const char * cublasv2GetErrorString(cublasStatus_t value) {

	switch (value) {
		case CUBLAS_STATUS_SUCCESS:
			return "No error";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS library not initialized";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "Alloc failed";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "Invalid value";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "Architecture mismatch";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "Mapping error";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "Execution failed";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "Internal error";
		default:
			return "Unknown error";
	}
}

const char * magmaGetErrorString(magma_err_t error) {
	return (magma_strerror(error));
}
