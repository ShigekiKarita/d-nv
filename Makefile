SUBDIRS := source

.PHONY: all $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@
