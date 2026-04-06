PITORCH_ROOT := $(abspath ../../..)

PROGS := test_generate.c test_inference_bench.c

COMMON_SRC := $(PITORCH_ROOT)/pitorch/pt.c \
              $(PITORCH_ROOT)/pitorch/model/llama2.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_ops.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_math.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec_tmu_shader.c \
              $(PITORCH_ROOT)/pitorch/runtime/arena.c \
              $(PITORCH_ROOT)/pitorch/text/pt_text.c \
              $(PITORCH_ROOT)/pitorch/train/pt_train.c \
              $(PITORCH_ROOT)/pitorch/train/pt_backward_ops.c

INC += -I$(PITORCH_ROOT)/pitorch \
       -I$(PITORCH_ROOT)/pitorch/model \
       -I$(PITORCH_ROOT)/pitorch/ops/core \
       -I$(PITORCH_ROOT)/pitorch/ops/matvec \
       -I$(PITORCH_ROOT)/pitorch/text \
       -I$(PITORCH_ROOT)/pitorch/train

LIBS += $(LGCC)

include ../../../pitorch.mk
include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
