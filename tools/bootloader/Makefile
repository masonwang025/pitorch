# Build both sides of the pitorch bootloader.
#   make my-install  — host-side UART loader (goes on your PATH or into bin/)
#   make kernel.img  — pi-side bootloader   (goes on the SD card)
#   make all         — both

all: my-install kernel.img

my-install:
	@make -C unix-side

kernel.img:
	@make -C pi-side

install: my-install
	@mkdir -p ../bin
	cp unix-side/my-install ../bin/my-install
	@echo "Installed my-install to bin/my-install"

clean:
	@make -C unix-side clean
	@make -C pi-side clean

.PHONY: all my-install kernel.img install clean
