################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/config.c \
../src/params.c 

CU_SRCS += \
../src/amica.cu \
../src/cudaamica.cu \
../src/device.cu \
../src/error.cu \
../src/helpers.cu \
../src/kernels.cu \
../src/preprocess.cu 

CU_DEPS += \
./src/amica.d \
./src/cudaamica.d \
./src/device.d \
./src/error.d \
./src/helpers.d \
./src/kernels.d \
./src/preprocess.d 

OBJS += \
./src/amica.o \
./src/config.o \
./src/cudaamica.o \
./src/device.o \
./src/error.o \
./src/helpers.o \
./src/kernels.o \
./src/params.o \
./src/preprocess.o 

C_DEPS += \
./src/config.d \
./src/params.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-6.0/bin/nvcc -DHAVE_CUBLAS -DUSELONG -DNORAND -DDEBUG=3 -I"/Volumes/Home/liaa/ica/cuda-workspace/CudAmica/include" -I/opt/intel/mkl/include -I/opt/magma/include -G -g -O0 -maxrregcount 32 -Xcompiler -fPIC -Xptxas -v -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 --target-cpu-architecture x86 -m64 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-6.0/bin/nvcc --compile -DHAVE_CUBLAS -DUSELONG -DNORAND -DDEBUG=3 -G -I"/Volumes/Home/liaa/ica/cuda-workspace/CudAmica/include" -I/opt/intel/mkl/include -I/opt/magma/include -O0 -Xcompiler -fPIC -Xptxas -v -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -maxrregcount 32 --target-cpu-architecture x86 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-6.0/bin/nvcc -DHAVE_CUBLAS -DUSELONG -DNORAND -DDEBUG=3 -I"/Volumes/Home/liaa/ica/cuda-workspace/CudAmica/include" -I/opt/intel/mkl/include -I/opt/magma/include -G -g -O0 -maxrregcount 32 -Xcompiler -fPIC -Xptxas -v -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 --target-cpu-architecture x86 -m64 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-6.0/bin/nvcc -DHAVE_CUBLAS -DUSELONG -DNORAND -DDEBUG=3 -I"/Volumes/Home/liaa/ica/cuda-workspace/CudAmica/include" -I/opt/intel/mkl/include -I/opt/magma/include -G -g -O0 -maxrregcount 32 -Xcompiler -fPIC -Xptxas -v --compile --target-cpu-architecture x86 -m64  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


