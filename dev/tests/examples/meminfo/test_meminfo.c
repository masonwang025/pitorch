#include "rpi.h"
#include "mailbox.h"

static void mbox_get_memory(uint32_t tag, uint32_t *base, uint32_t *size) {
    uint32_t msg[8] __attribute__((aligned(16))) = {
        8 * sizeof(uint32_t),
        0,
        tag,
        2 * sizeof(uint32_t),
        0,
        0, 0,
        0
    };
    if (!mbox_property(msg))
        panic("mailbox query failed for tag 0x%x\n", tag);
    *base = msg[5];
    *size = msg[6];
}

void notmain(void) {
    uint32_t arm_base, arm_size, vc_base, vc_size;

    mbox_get_memory(0x00010005, &arm_base, &arm_size);
    mbox_get_memory(0x00010006, &vc_base, &vc_size);

    printk("ARM mem: base=0x%x size=0x%x (%d MB)\n", arm_base, arm_size, arm_size >> 20);
    printk("VC  mem: base=0x%x size=0x%x (%d MB)\n", vc_base, vc_size, vc_size >> 20);
    printk("Total: %d MB\n", (arm_size + vc_size) >> 20);
}
