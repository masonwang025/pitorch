#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <termios.h>

#include "libunix.h"

int set_tty_to_8n1(int fd, unsigned speed, double timeout) {
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr (fd, &tty) != 0)
        panic("tcgetattr failed\n");
    memset (&tty, 0, sizeof tty);

    cfsetspeed(&tty, speed);

    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;
    assert(timeout < 100 && timeout > 0);
    tty.c_cc[VTIME] = (int)(timeout *10);

    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |=  CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_oflag &= ~OPOST;

    if(tcsetattr (fd, TCSANOW, &tty) != 0)
        panic("tcsetattr failed\n");
    return fd;
}
