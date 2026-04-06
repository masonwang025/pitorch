# Toolchain and build variable definitions.
# PITORCH_ROOT must be set before including this file.

ifndef PITORCH_ROOT
$(error PITORCH_ROOT is not set. Source pitorch.mk or set it in your environment.)
endif

ARM = arm-none-eabi
CC = $(ARM)-gcc
LD  = $(ARM)-ld
AS  = $(ARM)-as
AR = $(ARM)-ar
OD  = $(ARM)-objdump
OCP = $(ARM)-objcopy
LPP = $(PITORCH_ROOT)/libpi
LPI ?= $(LPP)/libpi.a
LGCC ?= $(shell $(CC) -print-libgcc-file-name)

START ?= $(LPP)/staff-start.o
DEFAULT_START := $(LPP)/staff-start.o
MEMMAP ?= $(LPP)/memmap

INC += -I. -I$(LPP)/include -I$(LPP)/ -I$(LPP)/src  -I$(LPP)/libc -I$(LPP)/staff-private
OPT_LEVEL ?= -Og

CFLAGS += -D__RPI__ $(OPT_LEVEL) -Wall -nostdlib -nostartfiles -ffreestanding -mcpu=arm1176jzf-s -mtune=arm1176jzf-s  -std=gnu99 $(INC) -ggdb -Wno-pointer-sign  -Werror  -Wno-unused-function -Wno-unused-variable

LDFLAGS += 

ifdef CFLAGS_EXTRA
CFLAGS += $(CFLAGS_EXTRA)
endif

CFLAGS += -mno-unaligned-access
CFLAGS += -mtp=soft

CPP_ASFLAGS =  -nostdlib -nostartfiles -ffreestanding   -Wa,--warn -Wa,--fatal-warnings -Wa,-mcpu=arm1176jzf-s -Wa,-march=armv6zk   $(INC)
