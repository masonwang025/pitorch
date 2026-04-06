#include <assert.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include "libunix.h"

#define _SVID_SOURCE
#include <dirent.h>

static const char *ttyusb_prefixes[] = {
    "ttyUSB",       // linux
    "ttyACM",       // linux
    "cu.SLAB_USB",  // mac os
    "cu.usbserial", // mac os
    0
};

static int filter(const struct dirent *d) {
    for (const char **p = ttyusb_prefixes; *p; p++)
        if (strncmp(d->d_name, *p, strlen(*p)) == 0)
            return 1;
    return 0;
}

char *find_ttyusb(void) {
    struct dirent **namelist;
    int n = scandir("/dev", &namelist, filter, alphasort);
    if (n < 0) sys_die(scandir, "scandir failed on /dev");
    if (n == 0) panic("no ttyusb device found\n");
    if (n > 1) panic("found %d ttyusb devices, expected 1\n", n);

    char *name = strdupf("/dev/%s", namelist[0]->d_name);
    free(namelist[0]);
    free(namelist);
    return name;
}

char *find_ttyusb_last(void) {
    struct dirent **namelist;
    int n = scandir("/dev", &namelist, filter, alphasort);
    if (n < 0) sys_die(scandir, "scandir failed on /dev");
    if (n == 0) panic("no ttyusb device found\n");

    int best = 0;
    time_t best_time = 0;
    for (int i = 0; i < n; i++) {
        char *path = strdupf("/dev/%s", namelist[i]->d_name);
        struct stat st;
        if (stat(path, &st) < 0)
            sys_die(stat, "stat failed on <%s>", path);
        if (i == 0 || st.st_mtime > best_time) {
            best_time = st.st_mtime;
            best = i;
        }
        free(path);
    }

    char *name = strdupf("/dev/%s", namelist[best]->d_name);
    for (int i = 0; i < n; i++) free(namelist[i]);
    free(namelist);
    return name;
}

char *find_ttyusb_first(void) {
    struct dirent **namelist;
    int n = scandir("/dev", &namelist, filter, alphasort);
    if (n < 0) sys_die(scandir, "scandir failed on /dev");
    if (n == 0) panic("no ttyusb device found\n");

    int best = 0;
    time_t best_time = 0;
    for (int i = 0; i < n; i++) {
        char *path = strdupf("/dev/%s", namelist[i]->d_name);
        struct stat st;
        if (stat(path, &st) < 0)
            sys_die(stat, "stat failed on <%s>", path);
        if (i == 0 || st.st_mtime < best_time) {
            best_time = st.st_mtime;
            best = i;
        }
        free(path);
    }

    char *name = strdupf("/dev/%s", namelist[best]->d_name);
    for (int i = 0; i < n; i++) free(namelist[i]);
    free(namelist);
    return name;
}
