PITORCH_ROOT := $(abspath ../../../..)

PROGS := test_dual_link_rank0.c test_dual_link_rank1.c test_dual_link_rank2.c

COMMON_SRC := $(PITORCH_ROOT)/pitorch/dist/transport/pt_link_gpio.c \
              $(PITORCH_ROOT)/pitorch/dist/protocol/pt_proto.c

INC += -I$(PITORCH_ROOT)/pitorch/dist/transport \
       -I$(PITORCH_ROOT)/pitorch/dist/protocol

LIBS += $(LGCC)

include $(PITORCH_ROOT)/pitorch.mk
RUN = 0
include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
