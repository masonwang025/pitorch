CC      = cc
CFLAGS  = -O2 -Wall -I../../pitorch/ops/core -I../../pitorch/model -I../../pitorch/text \
          -I../../pitorch/train -I../../pitorch/profiler -I../../pitorch
COMMON  = ../../pitorch/model/llama2.c ../../pitorch/ops/core/pt_ops.c ../../pitorch/ops/core/pt_math.c
TRACE   = ../../pitorch/profiler/trace.c
TRAIN   = ../../pitorch/train/pt_train.c ../../pitorch/train/pt_backward_ops.c $(TRACE)
PTCTX   = ../../pitorch/pt.c

generate: generate.c ../../pitorch/text/pt_text.c $(COMMON) $(TRAIN) $(PTCTX)
	$(CC) $(CFLAGS) -lm -o $@ $^

test_forward: test_forward.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^

test_stage_forward: test_stage_forward.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^

test_tokenizer: test_tokenizer.c ../../pitorch/text/pt_text.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^

test_generate_host: test_generate_host.c ../../pitorch/text/pt_text.c $(COMMON) $(TRAIN) $(PTCTX)
	$(CC) $(CFLAGS) -lm -o $@ $^

test_sampler: test_sampler.c ../../pitorch/text/pt_text.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^

test_train_forward: ../../demo/tests/train/test_forward_train.c $(COMMON) $(TRAIN)
	$(CC) $(CFLAGS) -lm -o $@ $^

test_train: test_train.c $(COMMON) $(TRAIN) $(PTCTX)
	$(CC) $(CFLAGS) -lm -o $@ $^

test_train_interactive: test_train_interactive.c ../../pitorch/text/pt_text.c $(COMMON) $(TRAIN) $(PTCTX)
	$(CC) $(CFLAGS) -lm -o $@ $^

test_train_profiled: test_train_profiled.c $(COMMON) $(TRAIN) $(PTCTX)
	$(CC) $(CFLAGS) -lm -o $@ $^

test_stage_train: test_stage_train.c $(COMMON) $(TRAIN) $(PTCTX)
	$(CC) $(CFLAGS) -lm -o $@ $^

test_stage_train_4pi: test_stage_train_4pi.c $(COMMON) $(TRAIN) $(PTCTX)
	$(CC) $(CFLAGS) -lm -o $@ $^

all: generate test_forward test_stage_forward test_tokenizer test_generate_host test_sampler test_train_forward test_train test_train_interactive test_train_profiled test_stage_train test_stage_train_4pi

clean:
	rm -f generate test_forward test_stage_forward test_tokenizer test_generate_host test_sampler test_train_forward test_train test_train_interactive test_train_profiled test_stage_train test_stage_train_4pi expected.txt

.PHONY: all clean
