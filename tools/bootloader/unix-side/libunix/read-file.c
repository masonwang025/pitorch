#include <assert.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "libunix.h"

void *read_file(unsigned *size, const char *name) {
    struct stat st;
    if (stat(name, &st) < 0)
        sys_die(stat, "stat failed on <%s>", name);
    unsigned n = st.st_size;

    void *buf = calloc(1, pi_roundup(n, 4));
    demand(buf, "calloc failed");

    int file = open(name, O_RDONLY);
    if (file < 0)
        sys_die(open, "open failed on <%s>", name);
    if (n > 0)
        read_exact(file, buf, n);
    close(file);

    *size = n;
    return buf;
}
