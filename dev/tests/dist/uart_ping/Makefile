PITORCH_ROOT := $(abspath ../../../..)

PROGS := test_ping_rank0.c test_ping_rank1.c

COMMON_SRC := $(PITORCH_ROOT)/pitorch/dist/transport/pt_link_uart.c \
              $(PITORCH_ROOT)/pitorch/dist/protocol/pt_proto.c

INC += -I$(PITORCH_ROOT)/pitorch/dist/transport \
       -I$(PITORCH_ROOT)/pitorch/dist/protocol

include $(PITORCH_ROOT)/pitorch.mk

# Override RUN after pitorch.mk (it sets RUN=1)
RUN = 0

include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
