RM=rm -rf
NVCC=/Developer/NVIDIA/CUDA-5.5/bin/nvcc
MAGMADIR=/opt/magma
INTELDIR=/opt/intel

MKLDIR=$(INTELDIR)/mkl

# All of the sources participating in the build are defined here
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


OBJS += \
./amica.o \
./config.o \
./cudaamica.o \
./device.o \
./error.o \
./helpers.o \
./kernels.o \
./params.o \
./preprocess.o 


DEFINES=-DITERTEST -DHAVE_CUBLAS -DUSELONG 

INCLUDE=-I../include -I$(MKLDIR)/include -I$(MAGMADIR)/include

USER_OBJS := $(MAGMADIR)/lib/libmagma.a $(INTELDIR)/lib/libiomp5.a $(MKLDIR)/lib/libmkl_intel_thread.a $(MKLDIR)/lib/libmkl_intel_lp64.a $(MKLDIR)/lib/libmkl_core.a

LIBS := -lcurand -lpthread -lcublas



CUDAFLAGS=-O2 -Xcompiler -fPIC -Xptxas -v -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -m64 -maxrregcount 32  -x cu 
CFLAGS=-O2 -maxrregcount 32 -m64 -Xcompiler -fPIC -Xptxas -v --compile  -x c 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../src/%.cu
	$(NVCC) --compile $(DEFINES) $(INCLUDE) $(CUDAFLAGS) -o  "$@" "$<"

%.o: ../src/%.c
	$(NVCC)  $(DEFINES)  $(INCLUDE) $(CFLAGS) -o  "$@" "$<"





# All Target
all: CudAmica 

# Tool invocations
CudAmica: $(OBJS) $(USER_OBJS)
	$(NVCC) --cudart static -m64 -Xlinker -no_compact_unwind -link -o  "CudAmica" $(OBJS) $(USER_OBJS) $(LIBS)


# Other Targets
clean:
	-$(RM) $(OBJS) CudAmica
	-@echo ' '


.PHONY: all clean dependents


