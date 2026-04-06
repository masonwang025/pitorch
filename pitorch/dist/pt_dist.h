#ifndef PT_DIST_H
#define PT_DIST_H

#include "pt_link_gpio.h"

/*
 * Distributed runtime context for multi-Pi pipeline.
 *
 * Every Pi in the ring has two GPIO links:
 *   downstream (high bank: 16-25) -> next rank
 *   upstream   (low bank:  4-13)  <- prev rank
 *
 * The caller sets l_start/l_end (layer assignment is test-specific).
 */
typedef struct {
    int rank, world_size;
    int l_start, l_end;     /* layer range [l_start, l_end) */
    int has_embed;          /* 1 if rank == 0 */
    int has_head;           /* 1 if rank == world_size - 1 */
    pt_link_gpio_t downstream;  /* high bank -> next */
    pt_link_gpio_t upstream;    /* low bank  <- prev */
} pt_dist_t;

/* Init GPIO links + flags. Does NOT set l_start/l_end -- caller does that. */
static inline pt_dist_t pt_dist_init_gpio(int rank, int world_size) {
    pt_dist_t d;
    d.rank = rank;
    d.world_size = world_size;
    d.l_start = 0;
    d.l_end = 0;
    d.has_embed = (rank == 0);
    d.has_head  = (rank == world_size - 1);
    d.downstream = pt_link_gpio_init(16, 24, 25);  /* high bank */
    d.upstream   = pt_link_gpio_init(4, 12, 13);    /* low bank */
    return d;
}

static inline void pt_dist_print(const pt_dist_t *d) {
    printk("rank %d/%d: layers [%d,%d) embed=%d head=%d\n",
           d->rank, d->world_size, d->l_start, d->l_end,
           d->has_embed, d->has_head);
}

#endif
