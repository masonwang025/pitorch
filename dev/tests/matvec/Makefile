PITORCH_ROOT := $(abspath ../../..)

PROGS := test_matvec.c

COMMON_SRC := $(PITORCH_ROOT)/pitorch/ops/matvec/matvec.c \
              $(PITORCH_ROOT)/pitorch/ops/matvec/matvec_tmu_shader.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_ops.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_math.c \
              $(PITORCH_ROOT)/pitorch/runtime/arena.c

INC += -I$(PITORCH_ROOT)/pitorch/ops/matvec \
       -I$(PITORCH_ROOT)/pitorch/ops/core

LIBS += $(LGCC)

include ../../../pitorch.mk
include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
