PITORCH_ROOT ?= $(abspath ..)
export PITORCH_ROOT

DEPS += ./Makefile ./manifest.mk
COMMON_SRC = $(SRC)

BUILD_DIR := ./objs
LIBNAME := libpi.a
START := ./staff-start.o

TTYUSB = 
GREP_STR := 'HASH:\|ERROR:\|PANIC:\|PASS:\|TEST:'
RUN = 1

DEPS += ./src

all:: ./staff-start.o

staff-start.o: $(BUILD_DIR)/staff-start.o
	cp $(BUILD_DIR)/staff-start.o .

include $(PITORCH_ROOT)/libpi/mk/Makefile.template-fixed

clean::
	rm -f staff-start.o
	rm -f staff-start-fp.o
	make -C  libc clean
	make -C  staff-src clean

.PHONY : libm test
