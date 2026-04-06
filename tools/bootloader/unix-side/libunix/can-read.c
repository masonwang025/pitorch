#include <assert.h>
#include <sys/select.h>
#include "libunix.h"

int can_read_timeout(int fd, unsigned usec) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = usec;

    int r;
    if((r = select(fd+1, &rfds, NULL, NULL, &tv)) < 0)
        sys_die(select, failed?);
    if(!FD_ISSET(fd, &rfds))
        return 0;
    return 1;
}

int can_read(int fd) { return can_read_timeout(fd, 0); }
