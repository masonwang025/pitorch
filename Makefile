TESTS := $(wildcard tests/*/.) $(wildcard tests/*/*/.)

.PHONY: all clean clean-tests clean-libpi clean-all

all:
	@echo "Usage:"
	@echo "  make clean        - remove all build artifacts (tests + libpi)"
	@echo "  make clean-tests  - remove test build artifacts only"
	@echo "  make clean-libpi  - remove libpi build artifacts only"
	@echo "  cd tests/<name> && bash run.sh  - build and run a test"

clean: clean-tests clean-libpi

clean-tests:
	@for t in $(TESTS); do \
		rm -rf $$t/objs $$t/*.bin $$t/*.list $$t/*_shader.c $$t/*_shader.h; \
	done
	@rm -f pitorch/ops/gemm/*_shader.c pitorch/ops/gemm/*_shader.h
	@echo "Cleaned test build artifacts."

clean-libpi:
	@make -s -C libpi clean 2>/dev/null || true
	@rm -f libpi/libpi.a.list
	@echo "Cleaned libpi build artifacts."

clean-all: clean
