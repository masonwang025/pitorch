PITORCH_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

INC += -I$(PITORCH_ROOT)/pitorch/runtime \
       -I$(PITORCH_ROOT)/pitorch/profiler \
       -I$(PITORCH_ROOT)/pitorch/ops/core

COMMON_SRC += $(PITORCH_ROOT)/pitorch/runtime/mailbox.c \
              $(PITORCH_ROOT)/pitorch/runtime/gpu.c \
              $(PITORCH_ROOT)/pitorch/runtime/mmu.c \
              $(PITORCH_ROOT)/pitorch/profiler/profiler.c \
              $(PITORCH_ROOT)/pitorch/profiler/trace.c

STAFF_OBJS += $(PITORCH_ROOT)/libpi/staff-objs/staff-kmalloc.o

BOOTLOADER = $(PITORCH_ROOT)/tools/bin/my-install
PI_RUN = $(PITORCH_ROOT)/tools/scripts/pi-run.sh
RUN = 1
