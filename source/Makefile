CUDA_DIR = $(shell dirname $(shell which nvcc))/..
CUDA_COMPILE_FLAGS = -Wall -Wextra -g3 -O0 -I $(CUDA_DIR)/samples/common/inc -std=c++11 # --gpu-architecture=compute_60 --gpu-code=sm_60,compute_60
CUDA_LINK_FLAGS = -lcuda -lnvrtc -lcudart -L $(CUDA_DIR)/lib64

DMD_COMPILE_FLAGS = -debug -g -gc -m64 -unittest -cov -J$(CUDA_DIR)/include
DMD_LIBDIR = $(shell dirname $(shell which dmd))/../lib64
DMD_LINK_FLAGS = -lphobos2 -L $(DMD_LIBDIR)

OBJECTS = cxxdriver.o driver.o compiler.o app.o typechecker.o parser.o

%.test: %.d
	rdmd --main $(DMD_COMPILE_FLAGS) $^

%.o: %.d
	dmd -c $^ $(DMD_COMPILE_FLAGS)

%.o: %.cpp
	g++ -c $^ $(CUDA_COMPILE_FLAGS)

app: $(OBJECTS) depend
	g++ -o $@ $(OBJECTS) $(CUDA_LINK_FLAGS) $(DMD_LINK_FLAGS)

all: app

clean:
	rm -rf *.o *.lst app

check: app
	@LD_LIBRARY_PATH=$(DMD_LIBDIR):$(LD_LIBRARY_PATH) ./app

coverage: check
	@find . -name "*.lst" -print | xargs -I{} sh -c \
	"grep -A1 -B1 -n --color=auto  0000000 {}; tail -n 1 {}" 2>&1 | tee ../coverage.txt

depend:
	dmd --version
	nvcc --version
	g++ --version

.PHONY: all clean check test depend coverage

