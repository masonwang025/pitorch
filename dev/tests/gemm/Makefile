PITORCH_ROOT := $(abspath ../../..)

PROGS := test_gemm.c
COMMON_SRC := $(PITORCH_ROOT)/pitorch/ops/gemm/gemm.c \
              $(PITORCH_ROOT)/pitorch/ops/gemm/gemm_tmu_shader.c

INC += -I$(PITORCH_ROOT)/pitorch/ops/gemm
LIBS += $(LGCC)

include ../../../pitorch.mk
include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
