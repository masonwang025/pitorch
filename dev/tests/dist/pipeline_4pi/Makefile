PITORCH_ROOT := $(abspath ../../../..)

PROGS := test_pipeline_4pi_rank0.c test_pipeline_4pi_rank1.c \
         test_pipeline_4pi_rank2.c test_pipeline_4pi_rank3.c

COMMON_SRC := $(PITORCH_ROOT)/pitorch/pt.c \
              $(PITORCH_ROOT)/pitorch/model/llama2.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_ops.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_math.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec_tmu_shader.c \
              $(PITORCH_ROOT)/pitorch/runtime/arena.c \
              $(PITORCH_ROOT)/pitorch/train/pt_train.c \
              $(PITORCH_ROOT)/pitorch/train/pt_backward_ops.c \
              $(PITORCH_ROOT)/pitorch/dist/transport/pt_link_gpio.c \
              $(PITORCH_ROOT)/pitorch/dist/protocol/pt_proto.c

INC += -I$(PITORCH_ROOT)/pitorch \
       -I$(PITORCH_ROOT)/pitorch/model \
       -I$(PITORCH_ROOT)/pitorch/ops/core \
       -I$(PITORCH_ROOT)/pitorch/ops/matvec \
       -I$(PITORCH_ROOT)/pitorch/train \
       -I$(PITORCH_ROOT)/pitorch/dist \
       -I$(PITORCH_ROOT)/pitorch/dist/transport \
       -I$(PITORCH_ROOT)/pitorch/dist/protocol

LIBS += $(LGCC)

include $(PITORCH_ROOT)/pitorch.mk
RUN = 0
include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
