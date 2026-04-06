PITORCH_ROOT := $(abspath ../../../..)

PROGS := test_baseline_single.c

COMMON_SRC := $(PITORCH_ROOT)/pitorch/pt.c \
              $(PITORCH_ROOT)/pitorch/model/llama2.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_ops.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_math.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec_tmu_shader.c \
              $(PITORCH_ROOT)/pitorch/runtime/arena.c \
              $(PITORCH_ROOT)/pitorch/train/pt_train.c \
              $(PITORCH_ROOT)/pitorch/train/pt_backward_ops.c

INC += -I$(PITORCH_ROOT)/pitorch \
       -I$(PITORCH_ROOT)/pitorch/model \
       -I$(PITORCH_ROOT)/pitorch/ops/core \
       -I$(PITORCH_ROOT)/pitorch/ops/matvec \
       -I$(PITORCH_ROOT)/pitorch/train

LIBS += $(LGCC)

include $(PITORCH_ROOT)/pitorch.mk

# Override RUN after pitorch.mk (it sets RUN=1)
RUN = 0

include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
