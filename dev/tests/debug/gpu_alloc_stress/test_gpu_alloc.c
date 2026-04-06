// TLDR:  the 3rd alloc fails even though prior allocations were freed

#include "rpi.h"

/* INLINE MAILBOX INTERFACE FOR REPRODUCIBILITY */

#define MBOX_BASE 0x2000B880
#define MBOX_READ (*(volatile uint32_t *)(MBOX_BASE + 0x0))
#define MBOX_STATUS (*(volatile uint32_t *)(MBOX_BASE + 0x18))
#define MBOX_WRITE (*(volatile uint32_t *)(MBOX_BASE + 0x20))
#define MBOX_FULL 0x80000000

static int mbox_prop(uint32_t *p)
{
    if ((uint32_t)p & 0xF)
        return 0;
    while (MBOX_STATUS & MBOX_FULL)
        ;
    MBOX_WRITE = ((uint32_t)p & ~0xF) | 8;
    while (1)
    {
        uint32_t r = MBOX_READ;
        if ((r & 0xF) == 8 && (r & ~0xF) == (uint32_t)p)
            break;
    }
    return (p[1] == 0x80000000);
}

/*  MAILBOX WRAPPERS */

static uint32_t do_alloc(uint32_t size)
{
    uint32_t p[9] __attribute__((aligned(16))) = {
        36, 0, 0x3000c, 12, 12, size, 4096, 0xC, 0};
    int ok = mbox_prop(p);
    printk("  alloc(%d): %s  handle=%d\n", size, ok ? "ok" : "FAIL", p[5]);
    return ok ? p[5] : 0;
}

static int do_lock(uint32_t h)
{
    uint32_t p[7] __attribute__((aligned(16))) = {28, 0, 0x3000d, 4, 4, h, 0};
    int ok = mbox_prop(p);
    printk("  lock(%d):   %s  bus=0x%x\n", h, ok ? "ok" : "FAIL", p[5]);
    return ok;
}

static int do_unlock(uint32_t h)
{
    uint32_t p[7] __attribute__((aligned(16))) = {28, 0, 0x3000e, 4, 4, h, 0};
    int ok = mbox_prop(p);
    printk("  unlock(%d): %s\n", h, ok ? "ok" : "FAIL");
    return ok;
}

static int do_free(uint32_t h)
{
    uint32_t p[7] __attribute__((aligned(16))) = {28, 0, 0x3000f, 4, 4, h, 0};
    int ok = mbox_prop(p);
    printk("  free(%d):   %s\n", h, ok ? "ok" : "FAIL");
    return ok;
}

/* ---- test ---- */

void notmain(void)
{
    printk("gpu mailbox alloc/free stress test\n");
    printk("testing sequential alloc/lock/unlock/free cycles:\n");

    // cycles 0 and 1 succeed while 2 fails

    for (int i = 0; i < 5; i++)
    {
        printk("cycle %d:\n", i);
        uint32_t h = do_alloc(4096);
        if (!h)
        {
            printk("  alloc failed!!!\n\n");
            break;
        }
        if (!do_lock(h))
        {
            printk("  ^^ lock failed\n\n");
            break;
        }
        if (!do_unlock(h))
        {
            printk("  ^^ unlock failed\n\n");
            break;
        }
        if (!do_free(h))
        {
            printk("  ^^ free failed\n\n");
            break;
        }
        printk("\n");
    }
}
