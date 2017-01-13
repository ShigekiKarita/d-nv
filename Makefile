SUBDIRS := source

.PHONY: all $(SUBDIRS) coverage clean
$(SUBDIRS):
	$(MAKE) -C $@

d-nvrtc: dub.json
	make -C source ../libcxxdriver.so
	dub run

coverage: d-nvrtc
	@find . -name "*.lst" -print | xargs -I{} sh -c \
	"grep -A1 -B1 -n --color=auto  0000000 {}; tail -n 1 {}" 2>&1 | tee ./coverage

clean:
	make -C source clean
	rm -rf *.lst *.so d-nvrtc

