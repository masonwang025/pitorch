# Builds kernel.img (the SD-card bootloader) against pitorch's vendored libpi.
PITORCH_ROOT := $(abspath ../../..)
include $(PITORCH_ROOT)/libpi/defs.mk

all: kernel.img

$(LPI): FORCE
	@make -C $(LPP)

boot-start.o: boot-start.S
	$(CC) $(CPP_ASFLAGS) -c $< -o $@

main.o: main.c get-code.h boot-defs.h boot-crc32.h
	$(CC) $(CFLAGS) -c $< -o $@

main.elf: boot-start.o main.o $(LPI)
	$(LD) boot-start.o main.o -T $(MEMMAP) -o $@ $(LPI) $(LGCC)

main.bin: main.elf
	$(OCP) $< -O binary $@

kernel.img: main.bin
	cp main.bin kernel.img

clean:
	rm -f *.o *.elf *.bin kernel.img

.PHONY: all clean FORCE
