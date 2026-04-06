CC = cc
CFLAGS = -Wall -g -Ilibunix -I../pi-side

LIBUNIX_SRC = $(wildcard libunix/*.c)
SRC = my-install.c put-code.c pi-echo.c set-tty-8n1.c $(LIBUNIX_SRC)
HDR = $(wildcard libunix/*.h) $(wildcard *.h) ../pi-side/boot-defs.h ../pi-side/boot-crc32.h

my-install: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC)

clean:
	rm -f my-install

.PHONY: clean
