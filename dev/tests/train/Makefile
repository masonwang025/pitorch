CC      = cc
CFLAGS  = -O2 -Wall -I../../../pitorch/ops/core -I../../../pitorch/model -I../../../pitorch/train -I../../../pitorch/profiler
COMMON  = ../../../pitorch/model/llama2.c ../../../pitorch/ops/core/pt_ops.c \
          ../../../pitorch/ops/core/pt_math.c ../../../pitorch/train/pt_train.c \
          ../../../pitorch/profiler/trace.c
BKWD    = ../../../pitorch/train/pt_backward_ops.c

test_forward_train: test_forward_train.c $(COMMON) $(BKWD)
	$(CC) $(CFLAGS) -lm -o $@ $^

test_backward_ops: test_backward_ops.c $(BKWD) ../../../pitorch/ops/core/pt_math.c
	$(CC) $(CFLAGS) -lm -o $@ $^

backward_refs.bin: gen_backward_refs.py
	python3 gen_backward_refs.py

all: test_forward_train test_backward_ops

check_backward: test_backward_ops backward_refs.bin
	./test_backward_ops backward_refs.bin

clean:
	rm -f test_forward_train test_backward_ops expected.txt backward_refs.bin

.PHONY: all clean check_backward
