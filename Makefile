CC ?= gcc
CUDA_HOME ?= /opt/cuda
CUDA_LIBDIR_DEFAULT := $(CUDA_HOME)/lib64
CUDA_LIBDIR ?= $(CUDA_LIBDIR_DEFAULT)
CUDA_ARCH ?= $(shell uname -m)
CUDA_LIBDIR_ALT := $(CUDA_HOME)/targets/$(CUDA_ARCH)-linux/lib
OPENMP ?= 1

ifeq ($(CUDA_LIBDIR),$(CUDA_LIBDIR_DEFAULT))
  ifeq ($(wildcard $(CUDA_LIBDIR)/libnvptxcompiler_static.a)$(wildcard $(CUDA_LIBDIR)/libnvptxcompiler.so),)
    ifneq ($(wildcard $(CUDA_LIBDIR_ALT)/libnvptxcompiler_static.a)$(wildcard $(CUDA_LIBDIR_ALT)/libnvptxcompiler.so),)
      CUDA_LIBDIR := $(CUDA_LIBDIR_ALT)
    endif
  endif
endif

CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -I. -I$(CUDA_HOME)/include
DEFS ?= -DPTX_INJECT_MAX_UNIQUE_INJECTS=16384
CFLAGS += $(DEFS)

LDFLAGS ?=
NVPTXCOMPILER_LIB ?= $(if $(wildcard $(CUDA_LIBDIR)/libnvptxcompiler_static.a),nvptxcompiler_static,nvptxcompiler)
LDLIBS = -L$(CUDA_LIBDIR) -lnvrtc -l$(NVPTXCOMPILER_LIB) -lm -ldl -lpthread

ifeq ($(NVPTXCOMPILER_LIB),nvptxcompiler_static)
  LDLIBS += -lstdc++
endif

ifeq ($(OPENMP),1)
  CFLAGS += -fopenmp
  LDLIBS += -fopenmp
endif

SRC = stack_ptx_cubin_bench_standalone.c stack_ptx_inject_kernel_gen.c
OBJ = $(SRC:.c=.o)
BIN = stack_ptx_ptx_inject_bench

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean
