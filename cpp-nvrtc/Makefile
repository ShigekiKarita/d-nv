CUDA_DIR = $(shell dirname $(shell which nvcc))/..
CUDA_COMPILE_FLAGS = -I $(CUDA_DIR)/samples/common/inc
CUDA_LINK_FLAGS = -lcuda -lnvrtc

DMD_LIBDIR = $(shell dirname $(shell which dmd))/../lib64
DMD_LINK_FLAGS = -lphobos2 -L $(DMD_LIBDIR)

OBJECTS = app.o vectorAdd.o

%.o: %.d
	dmd -c $^

%.o: %.cpp
	nvcc -c $^ $(CUDA_COMPILE_FLAGS)

app: $(OBJECTS)
	g++ -o $@ $(OBJECTS) $(CUDA_LINK_FLAGS) $(DMD_LINK_FLAGS)

all: app

clean:
	rm -rf *.o app

check: app
	@LD_LIBRARY_PATH=$(DMD_LIBDIR):$(LD_LIBRARY_PATH) ./app

.PHONY: all clean check
