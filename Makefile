SUBDIRS := source/dnv
CUDA_DIR = $(shell dirname $(shell which nvcc))/..
CUDA_COMPILE_FLAGS = -Wall -Wextra -g3 -O0 -I $(CUDA_DIR)/samples/common/inc -std=c++11 # --gpu-architecture=compute_60 --gpu-code=sm_60,compute_60
CUDA_LINK_FLAGS = -lcuda -lnvrtc -lcudart -L $(CUDA_DIR)/lib64


.PHONY: all $(SUBDIRS) coverage clean coveralls
$(SUBDIRS):
	$(MAKE) -C $@

libcxxdriver.so: $(SUBDIRS)/cxxdriver.cpp
	g++ $^ $(CUDA_COMPILE_FLAGS) -fPIC -shared -o $@

d-nvrtc: dub.json libcxxdriver.so
	dub --build=unittest-cov

coverage: d-nvrtc
	@find . -name "*.lst" -print | xargs -I{} sh -c \
	"grep -A1 -B1 -n --color=auto  0000000 {}; tail -n 1 {}" 2>&1 | tee ./coverage

coveralls:
	dub test -b unittest-cov
	dub run doveralls -- -t ms4TIpE3i9sXyM9JYivIPSM5BjdY957Ob

clean:
	dub clean
	make -C $(SUBDIRS) clean
	rm -rf *.lst *.so d-nvrtc

