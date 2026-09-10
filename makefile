CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC
PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)

NVCC = nvcc
ARCH = -arch=sm_75
CUDA_FLAGS = -lcurand -lcuda

NPOINTS ?=
FPS ?=
SECONDS ?=

.PHONY: all deps clean run

all: energy_kernel.cubin generate octree_lib.so

deps:
	pip install pycuda
	pip install numpy
	pip install pybind11

energy_kernel.cubin: energy_kernel.cu
	$(NVCC) $(ARCH) -cubin $< -o $@

generate: generate.cu
	$(NVCC) $(ARCH) $< -o $@ $(CUDA_FLAGS)

octree_lib.so: octree.cpp
	$(CC) $(CFLAGS) $(PYBIND_INCLUDES) $< -o $@

run:
	./generate $(NPOINTS) | python3 simulate.py | python3 viewer.py $(SECONDS) $(FPS)

clean:
	rm -f energy_kernel.cubin generate octree_lib.so