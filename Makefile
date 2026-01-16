CC ?= gcc
CUDA_HOME ?= /opt/cuda
CUDA_LIBDIR ?= $(CUDA_HOME)/lib64
OPENMP ?= 1

CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -I. -I$(CUDA_HOME)/include
DEFS ?= -DPTX_INJECT_MAX_UNIQUE_INJECTS=16384
CFLAGS += $(DEFS)

LDFLAGS ?=
NVPTXCOMPILER_LIB ?= $(if $(wildcard $(CUDA_LIBDIR)/libnvptxcompiler_static.a),nvptxcompiler_static,nvptxcompiler)
LDLIBS = -L$(CUDA_LIBDIR) -lnvrtc -l$(NVPTXCOMPILER_LIB) -lm -ldl -lpthread

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
