#include "rpi.h"
#include "pt_proto.h"
#include "our-crc32.h"

int pt_proto_send(pt_transport_t *t, uint32_t opcode, const void *payload, unsigned len) {
    pt_proto_hdr_t hdr;
    hdr.magic  = PT_PROTO_MAGIC;
    hdr.opcode = opcode;
    hdr.len    = len;
    hdr.crc32  = len > 0 ? our_crc32(payload, len) : 0;

    t->send_raw(t, &hdr, sizeof(hdr));
    if (len > 0)
        t->send_raw(t, payload, len);
    return 0;
}

int pt_proto_recv(pt_transport_t *t, uint32_t *opcode, void *buf,
                  uint32_t max_len, uint32_t *payload_len) {
    pt_proto_hdr_t hdr;

    if (t->recv_raw(t, &hdr, sizeof(hdr)) < 0) {
        printk("pt_proto_recv: timeout on header\n");
        return -1;
    }

    if (hdr.magic != PT_PROTO_MAGIC) {
        printk("pt_proto_recv: bad magic 0x%x (expected 0x%x)\n",
               hdr.magic, PT_PROTO_MAGIC);
        return -1;
    }

    if (hdr.len > max_len) {
        printk("pt_proto_recv: payload %d > buf %d\n", hdr.len, max_len);
        return -1;
    }

    if (hdr.len > 0) {
        if (t->recv_raw(t, buf, hdr.len) < 0) {
            printk("pt_proto_recv: timeout on payload\n");
            return -1;
        }

        uint32_t computed = our_crc32(buf, hdr.len);
        if (computed != hdr.crc32) {
            printk("pt_proto_recv: CRC mismatch: got 0x%x expected 0x%x\n",
                   computed, hdr.crc32);
            return -1;
        }
    }

    *opcode = hdr.opcode;
    *payload_len = hdr.len;
    return 0;
}
