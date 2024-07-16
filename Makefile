GIT_HASH := $(shell git rev-parse HEAD)
GIT_MODIFIED := $(shell git status --porcelain)

GIT_FLAGS := -DGIT_HASH=$(GIT_HASH) -DGIT_MODIFIED="$(GIT_MODIFIED)"

FGPM_BUILD_DIR ?= build
FGPM_SRC_DIR ?= src
FGPM_INCLUDE ?= -I$(FGPM_SRC_DIR) -Iinclude
FGPM_DRIVERS ?= drivers

CUDA_DIR ?= /usr/local/cuda
CUDA_ARCH_FLAGS ?= -arch=sm_60 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
CUDA_CC ?= nvcc

# MPI_CXX ?= mpicxx
MPI_CXX ?= g++

MPI_OBJECT_FLAGS ?= -std=c++17 -I$(CUDA_DIR)/include $(FGPM_INCLUDE) -fPIC -O3 -fopenmp -g -Wall -Wpedantic -Werror
NVCC_OBJECT_FLAGS ?= -std=c++17 -lcufft -lineinfo -Xptxas -v -Xcompiler="-fPIC,-O3,-fopenmp,-g,-Wall,-Werror" $(CUDA_ARCH_FLAGS) $(FGPM_INCLUDE)

CUDA_LINK_FLAGS ?= -L$(CUDA_DIR)/lib64 -lcudart -lcufft

FGPM_SOURCES := $(shell find $(FGPM_SRC_DIR) -name '*.cpp') $(shell find $(FGPM_SRC_DIR) -name '*.cu')
FGPM_OBJECTS_1 := $(FGPM_SOURCES:%.cpp=%.o)
FGPM_OBJECTS := $(FGPM_OBJECTS_1:%.cu=%.o)
FGPM_OUTPUTS := $(FGPM_OBJECTS:%=build/%)

FGPM_DRIVER_SOURCES := $(shell find $(FGPM_DRIVERS) -name '*.cpp')
FGPM_DRIVER_OBJECTS := $(FGPM_DRIVER_SOURCES:%.cpp=build/%.o)

FGPM_INCLUDE_FILES := $(shell find $(FGPM_SRC_DIR) -name '*.hpp') $(shell find include -name '*.hpp')

main: build/driver

.secondary: $(FGPM_OUTPUTS) $(FGPM_DRIVER_OBJECTS)

$(FGPM_BUILD_DIR)/%: $(FGPM_BUILD_DIR)/$(FGPM_DRIVERS)/%.o $(FGPM_OUTPUTS)
	echo $(FGPM_DRIVER_OBJECTS)
	mkdir -p $(@D)
	$(MPI_CXX) $^ $(MPI_OBJECT_FLAGS) $(CUDA_LINK_FLAGS) $(GIT_FLAGS) -o $@

$(FGPM_BUILD_DIR)/%.o: %.cu $(FGPM_INCLUDE_FILES)
	mkdir -p $(@D)
	$(CUDA_CC) -c $< $(NVCC_OBJECT_FLAGS) $(GIT_FLAGS) -o $@

$(FGPM_BUILD_DIR)/%.o: %.cpp $(FGPM_INCLUDE_FILES)
	mkdir -p $(@D)
	$(MPI_CXX) -c $< $(MPI_OBJECT_FLAGS) $(GIT_FLAGS) -o $@

.PHONY: clean
clean:
	rm -rf $(FGPM_BUILD_DIR)