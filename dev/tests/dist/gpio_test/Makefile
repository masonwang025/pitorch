PITORCH_ROOT := $(abspath ../../../..)

PROGS := test_gpio_tx.c test_gpio_rx.c test_byte_tx.c test_byte_rx.c \
         test_gpio_echo_rank0.c test_gpio_echo_rank1.c \
         test_pin_sweep_rank0.c test_pin_sweep_rank1.c \
         test_clk_ack_rank0.c test_clk_ack_rank1.c \
         test_bulk_rank0.c test_bulk_rank1.c \
         test_chunked_rank0.c test_chunked_rank1.c

COMMON_SRC := $(PITORCH_ROOT)/pitorch/dist/transport/pt_link_uart.c \
              $(PITORCH_ROOT)/pitorch/dist/transport/pt_link_gpio.c \
              $(PITORCH_ROOT)/pitorch/dist/protocol/pt_proto.c

INC += -I$(PITORCH_ROOT)/pitorch/dist/transport \
       -I$(PITORCH_ROOT)/pitorch/dist/protocol

LIBS += $(LGCC)

include $(PITORCH_ROOT)/pitorch.mk
RUN = 0
include $(PITORCH_ROOT)/libpi/mk/Makefile.robust-v2
