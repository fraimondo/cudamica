AUTOMAKE_OPTIONS = foreign

bin_PROGRAMS = cudaamica
SRC=src
INCLUDE=-Iinclude/ -I/opt/intel/mkl/include -I/opt/magma-1.2.1/include

EXTRA_DIST=COPYING
SRCINCLUDE=include/
noinst_HEADERS = $(SRCINCLUDE)/config.h $(SRCINCLUDE)/amica.h $(SRCINCLUDE)/device.h $(SRCINCLUDE)/error.h $(SRCINCLUDE)/helpers.h $(SRCINCLUDE)/kernels.h $(SRCINCLUDE)/params.h $(SRCINCLUDE)/preprocess.h $(SRCINCLUDE)/tipos.h $(SRCINCLUDE)/tools.h
cudaamica_SOURCES = $(SRC)/amica.cu $(SRC)/config.c $(SRC)/device.cu $(SRC)/error.cu $(SRC)/helpers.cu $(SRC)/kernels.cu $(SRC)/params.c $(SRC)/preprocess.cu
cudaamica_CFLAGS = -Iinclude/ -I/opt/intel/mkl/include -I/opt/magma-1.2.1/include $(CUDA_CFLAGS)
SUFFIXES=.cu .c
FEDE_LIBS=/opt/magma-1.2.1/lib/libmagma.a /opt/intel/lib/libiomp5.a /opt/intel/mkl/lib/libmkl_intel_thread.a /opt/intel/mkl/lib/libmkl_intel_lp64.a /opt/intel/mkl/lib/libmkl_core.a /opt/magma-1.2.1/lib/libmagmablas.a
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $(LIBS) $(CUDA_LIBS) -c -o $@ $<

.c.o:
	$(CC) $(INCLUDE) $(LIBS) -c -o $@ $<
	
	
cudaamica$(EXEEXT): $(SRC)/cudaamica.cu $(cudaamica_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $(LIBS) $(CUDA_LIBS) $(FEDE_LIBS) $(CUDAAMICA_NVCCFLAGS) $(PTHREAD_LIBS) -o $@ $^
	
dist-hook:
	mkdir -p .tmpclean
	rm -rf .tmpclean/*
	cp -r $(distdir)/* .tmpclean/
	./deleter.py .tmpclean/ $(distdir)/
	rm -rf .tmpclean
	echo "echo $(shell ./evalrev) " > $(distdir)/evalrev
	chmod +x $(distdir)/evalrev