#ifndef __LIBUNIX_H__
#define __LIBUNIX_H__
// Minimal libunix for the pitorch bootloader.
// Vendored from cs140e with unused functions removed.
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

#define pi_roundup(x,n) (((x)+((n)-1))&(~((n)-1)))

#include "demand.h"

int pi_done(unsigned char *s);
void remove_nonprint(uint8_t *buf, int n);

void *read_file(unsigned *size, const char *name);
int open_tty(const char *device);
int set_tty_to_8n1(int fd, unsigned speed, double timeout);
int tty_gone(const char *ttyname);
int exists(const char *name);

int write_exact(int fd, const void *data, unsigned n);
int read_exact(int fd, void *data, unsigned n);

void put_uint8(int fd, uint8_t b);
void put_uint32(int fd, uint32_t u);
uint8_t get_uint8(int fd);
uint32_t get_uint32(int fd);

int suffix_cmp(const char *s, const char *suffix);
int prefix_cmp(const char *s, const char *prefix);

void pi_echo(int unix_fd, int pi_fd, const char *portname);

char *find_ttyusb(void);
char *find_ttyusb_first(void);
char *find_ttyusb_last(void);

#include "bit-support.h"

int can_read_timeout(int fd, unsigned usec);
int can_read(int fd);
int read_timeout(int fd, void *data, unsigned n, unsigned timeout);

void argv_print(const char *msg, char *argv[]);

char *strcatf(char *dst, const char *fmt, ...);
char *strcpyf(char *dst, const char *fmt, ...);
char *strdupf(const char *fmt, ...);
char *vstrdupf(const char *fmt, va_list ap);
char *str2dupf(const char *src1, const char *fmt, ...);

#define gcc_mb() asm volatile ("" : : : "memory")

#endif
