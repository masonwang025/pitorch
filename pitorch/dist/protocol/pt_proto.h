#ifndef PT_PROTO_H
#define PT_PROTO_H

#include "pt_transport.h"

#define PT_PROTO_MAGIC  0x50540001   /* "PT" + version 1 */

enum {
    PT_OP_PING  = 1,
    PT_OP_PONG  = 2,
    PT_OP_DATA  = 3,
    PT_OP_ACK   = 4,
};

typedef struct {
    uint32_t magic;
    uint32_t opcode;
    uint32_t len;       /* payload length in bytes */
    uint32_t crc32;     /* CRC32 of payload (0 if len == 0) */
} pt_proto_hdr_t;

/* Send a framed message: header + payload. */
int pt_proto_send(pt_transport_t *t, uint32_t opcode, const void *payload, unsigned len);

/* Receive a framed message. Verifies magic and CRC.
 * buf must be at least max_len bytes.
 * On success: fills *opcode, *payload_len, returns 0.
 * On error: returns -1. */
int pt_proto_recv(pt_transport_t *t, uint32_t *opcode, void *buf,
                  uint32_t max_len, uint32_t *payload_len);

#endif
