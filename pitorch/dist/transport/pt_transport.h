#ifndef PT_TRANSPORT_H
#define PT_TRANSPORT_H

/*
 * Abstract transport layer for inter-Pi communication.
 * Both sw-UART and GPIO parallel bus implement this interface.
 * The protocol layer (pt_proto) uses this to stay transport-agnostic.
 */

typedef struct pt_transport pt_transport_t;

typedef void (*pt_raw_send_fn)(pt_transport_t *t, const void *buf, unsigned len);
typedef int  (*pt_raw_recv_fn)(pt_transport_t *t, void *buf, unsigned len);

struct pt_transport {
    pt_raw_send_fn send_raw;
    pt_raw_recv_fn recv_raw;
};

#endif
