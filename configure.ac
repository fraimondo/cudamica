#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([cudaica], [2.0], [fraimondo@dc.uba.ar])
AM_INIT_AUTOMAKE([cudaamica], [2.0])
#AC_CONFIG_SRCDIR([cudaamica.c])
#~ AC_CONFIG_HEADERS([includes/config.h])
m4_include([m4/m4_ax_check_compile_flag.m4])
m4_include([m4/m4_ax_gcc_version.m4])
m4_include([m4/m4_ax_nvcc_version.m4])
m4_include([m4/m4_ax_cuda_version.m4])
m4_include([m4/m4_ax_pthreads.m4])
# Checks for programs.
AC_PROG_CXX([g++])
AC_PROG_CC([gcc])
AX_GCC_VERSION
ACX_PTHREAD
#~ echo $GCC_VERSION

AC_CANONICAL_HOST
# Check for which host we are on and setup a few things
# specifically based on the host
case $host_os in
  darwin* )
        OS=mac
        ;;
  linux* )
        OS=linux
        ;;
     *)
        #Default Case
        AC_MSG_ERROR([Your platform is not currently supported])
        ;;
esac

#CHECK FOR CUDA
AC_ARG_WITH([cuda],
    [AS_HELP_STRING([--with-cuda],
		[PATH prefix where cuda is installed  @<:@default=/usr/local/cuda@:>@])
	])
	
AC_ARG_WITH([cuda-arch],
    [AS_HELP_STRING([--with-cuda-arch],
		[CUDA Architecture to use  @<:@default=20@:>@. Options are:
		11 = Compute Capability 1.1 or above - 
		12 = Compute Capability 1.2 or above - 
		13 = Compute Capability 1.3 or above -
		20 = Compute Capability 2.0 or above -
		21 = Compute Capability 2.1 or above])],
	[CUDA_ARCH=$with_cuda_arch],
	[CUDA_ARCH=20])
    
AC_ARG_ENABLE([debug],
	[AS_HELP_STRING([--enable-debug],[Enables debugging features @<:@default=disabled@:>@])],
	[USEDEBUG="true" 
	DEBUG=$enable_debug],[USEDEBUG="false"])

AC_ARG_WITH([double],
	[AS_HELP_STRING([--with-double],[Enables double precission features @<:@default=yes@:>@])],
	[
	if test x$enable_double = xyes
	then
		if test $CUDA_ARCH -ge 13
		then
			USESINGLE="false"
		else
			AC_MSG_ERROR([Architecture 1.3 or higher needed for double precission floating point to be available.])
		fi
	else
		USESINGLE="true"
	fi
	],[if test $CUDA_ARCH -ge 13
		then
			USESINGLE="false"
		else
			AC_MSG_ERROR([Architecture 1.3 or higher needed for double precission floating point to be available.])
		fi
	])



AC_CHECK_SIZEOF([long])
AS_IF([test $ac_cv_sizeof_long -eq 8],
[OSBIT=64],
[OSBIT=32])

#~ echo "Architecture ${with_cuda_arch} ${CUDA_ARCH}"
#~ echo "BITS: ${OSBIT}"
working_nvcc=no
if test -z "$with_cuda"
then
	CUDA_DIR="/usr/local/cuda"
else
	CUDA_DIR="$with_cuda"
fi
CUDA_CFLAGS="-I$CUDA_DIR/include"
if test x$OSBIT = x64
then
	if test x$OS = xmac
	then 
		CUDA_LIBS="-L$CUDA_DIR/lib -lcudart -lcublas -lcurand"
	else
		CUDA_LIBS="-L$CUDA_DIR/lib64 -lcudart -lcublas -lcurand"
	fi
	OSFLAGS="-m64"
else
	CUDA_LIBS="-L$CUDA_DIR/lib -lcudart -lcublas -lcurand"
	OSFLAGS="-m32"
fi

AC_ARG_WITH([long],
	[AS_HELP_STRING([--with-long],[Enables use of long integers (64 bits) features @<:@default=yes@:>@])],
	[
	if test x$enable_long = xyes
	then
		USELONG="true"
		if test x$OSBIT != x64
		then
			AC_MSG_WARN([Using 64 bits integers on a 32 bits OS.])
		fi
	else
		USELONG="false"
	fi
	],[	USELONG="true"
		if test x$OSBIT != x64
		then
			AC_MSG_WARN([Using 64 bits integers on a 32 bits OS.])
		fi
	])


if test x$USEDEBUG = xtrue
then
    NVCCFLAGS="-DDEBUG=${DEBUG} -g -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} --ptxas-options=\"-v\" "
else
    NVCCFLAGS="-O -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} "
fi

if test x$USESINGLE = xtrue
then
	NVCCFLAGS+=" -DUSESINGLE"
fi

if test x$USELONG = xtrue
then
	NVCCFLAGS+=" -DUSELONG"
	CFLAGS+=" -DUSELONG"
fi

if test x$OS = xmac
then 
	CFLAGS+=" -no-compat-unwind"
	CUDAAMICA_NVCCFLAGS="-Xlinker -no_compact_unwind"
else
	CUDAAMICA_NVCCFLAGS=""
fi

VERSION="`./evalrev`"
NVCCFLAGS+=" -DSVNREVISION=${VERSION}"
NVCCFLAGS+=" ${OSFLAGS}"

AC_SUBST(CUDA_DIR)
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCCFLAGS)
AC_SUBST(CUDAAMICA_NVCCFLAGS)
AC_SUBST(CFLAGS)

#~ echo "$CUDA_DIR"
AC_PATH_PROG([NVCC], [nvcc], [], [$CUDA_DIR/bin])
#~ AC_CHECK_PROG([NVCC_EXISTS], [nvcc], [yes], [no], [$CUDA_DIR])
#~ echo "$NVCC"
if test "x$NVCC" != "x"
then
	AX_NVCC_VERSION
	AX_CUDA_VERSION
	NVCCFLAGS+=" -DCUDA_VERSION=${CUDA_VERSION}"
	NVCC_VERSION_MAJOR=$(echo $NVCC_VERSION | cut -d'.' -f1)
	NVCC_VERSION_MINOR=$(echo $NVCC_VERSION | cut -d'.' -f2)
	
	if test $NVCC_VERSION_MAJOR -lt 4
	then
		AC_MSG_ERROR([CUDA compiler version is lower than 4.0 = $NVCC_VERSION])
	fi
	
	AC_MSG_CHECKING([whether nvcc works])
	cat > conftest.cu <<EOF
#ifdef USESINGLE
			void __global__ whatever(float *c) {
			float a = 3.2;
			float b = 1.2;
#else
			void __global__ whatever(double *c) {
			double a = 3.2;
			double b = 1.2;
#endif
			*c = a * b;
			__syncthreads();

		}
		
		int main(int argc, char** argv)  {
#ifdef USESINGLE
			float a = 0.0;
#else
			double a = 0.0;
#endif
			whatever<<<1,1>>>(&a);
			return 0;
		}
EOF
	#~ echo "${NVCC} ${NVCCFLAGS} conftest.cu"
	if $NVCC $NVCCFLAGS conftest.cu
	then
		working_nvcc=yes
		AC_MSG_RESULT([Ok])
	else 
		working_nvcc=no
		AC_MSG_ERROR([CUDA compiler is not working right.])
	fi
	rm -f conftest.cu conftest.o
	
else
	AC_MSG_ERROR([CUDA compiler is needed.])
fi

#~ echo "NVCC= ${NVCC}"
#~ echo "NVCC FLAGS = ${NVCCFLAGS}"


# Checks for libraries.
# Checks for header files.
AC_CHECK_HEADERS([fcntl.h limits.h stdlib.h string.h unistd.h])
# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([dup2 memset strstr])

GCC_VERSION_MAJOR=$(echo $GCC_VERSION | cut -d'.' -f1)
GCC_VERSION_MINOR=$(echo $GCC_VERSION | cut -d'.' -f2)
GCC_VERSION_PATCH=$(echo $GCC_VERSION | cut -d'.' -f3)

if test $GCC_VERSION_MAJOR -eq 4
then
	if test $GCC_VERSION_MINOR -gt 4
	then
		if test $CUDA_VERSION -eq 4
		then
		AC_MSG_WARN([
=====================================================================
GCC Version is > 4.4 and NVCC compiler may not work. Please check
http://liaa.dc.uba.ar/?q=node/12
for more information on how to fix this issue.
=====================================================================])
		fi
	fi
fi
 

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
