SUBDIRS := source

.PHONY: all $(SUBDIRS) coverage clean coveralls
$(SUBDIRS):
	$(MAKE) -C $@

libcxxdriver.so: source/cxxdriver.cpp
	make -C source ../$@

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
	make -C source clean
	rm -rf *.lst *.so d-nvrtc

