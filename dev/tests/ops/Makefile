PITORCH_ROOT := $(abspath ../../..)

PROGS := test_ops.c
COMMON_SRC := $(PITORCH_ROOT)/pitorch/runtime/arena.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_math.c \
              $(PITORCH_ROOT)/pitorch/ops/core/pt_ops.c

INC += -I$(PITORCH_ROOT)/pitorch/ops/core
LIBS += $(LGCC)

include ../../../pitorch.mk
include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
