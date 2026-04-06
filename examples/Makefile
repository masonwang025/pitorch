PITORCH_ROOT := $(abspath ..)

# ── All demo programs ──
# Single-Pi demos + distributed rank shims (4 ranks × 2 demos)
PROGS := generate.c \
         train.c \
         ranks/train-distributed-rank0.c \
         ranks/train-distributed-rank1.c \
         ranks/train-distributed-rank2.c \
         ranks/train-distributed-rank3.c \
         ranks/generate-distributed-rank0.c \
         ranks/generate-distributed-rank1.c \
         ranks/generate-distributed-rank2.c \
         ranks/generate-distributed-rank3.c

# ── PiTorch sources (superset — covers single + distributed + text) ──
COMMON_SRC := $(PITORCH_ROOT)/pitorch/pt.c \
              $(PITORCH_ROOT)/pitorch/model/llama2.c \
              $(PITORCH_ROOT)/pitorch/model/pt_shard.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_ops.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_math.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec_tmu_shader.c \
              $(PITORCH_ROOT)/pitorch/ops/gemm/gemm_rect.c \
              $(PITORCH_ROOT)/pitorch/ops/gemm/gemm_rect_tmu_shader.c \
              $(PITORCH_ROOT)/pitorch/runtime/arena.c \
              $(PITORCH_ROOT)/pitorch/text/pt_text.c \
              $(PITORCH_ROOT)/pitorch/train/pt_train.c \
              $(PITORCH_ROOT)/pitorch/train/pt_backward_ops.c \
              $(PITORCH_ROOT)/pitorch/dist/transport/pt_link_gpio.c \
              $(PITORCH_ROOT)/pitorch/dist/protocol/pt_proto.c \
              $(PITORCH_ROOT)/pitorch/dist/pt_dist_pipeline.c

# ── Include paths ──
INC += -I$(PITORCH_ROOT)/pitorch \
       -I$(PITORCH_ROOT)/pitorch/model \
       -I$(PITORCH_ROOT)/pitorch/ops/core \
       -I$(PITORCH_ROOT)/pitorch/ops/matvec \
       -I$(PITORCH_ROOT)/pitorch/ops/gemm \
       -I$(PITORCH_ROOT)/pitorch/text \
       -I$(PITORCH_ROOT)/pitorch/train \
       -I$(PITORCH_ROOT)/pitorch/dist \
       -I$(PITORCH_ROOT)/pitorch/dist/transport \
       -I$(PITORCH_ROOT)/pitorch/dist/protocol

LIBS += $(LGCC)

include $(PITORCH_ROOT)/pitorch.mk
RUN = 0
include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
