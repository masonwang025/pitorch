#ifndef __PI_BITSUPPORT_H__
#define __PI_BITSUPPORT_H__

static inline uint32_t bit_clr(uint32_t x, unsigned bit) {
    assert(bit<32);
    return x & ~(1<<bit);
}
static inline uint32_t bit_set(uint32_t x, unsigned bit) {
    assert(bit<32);
    return x | (1<<bit);
}
static inline unsigned bit_is_on(uint32_t x, unsigned bit) {
    assert(bit<32);
    return (x >> bit) & 1;
}
#define bit_isset bit_is_on
#define bit_get bit_is_on

static inline unsigned bit_is_off(uint32_t x, unsigned bit) {
    return bit_is_on(x,bit) == 0;
}
static inline uint32_t bits_mask(unsigned nbits) {
    if(nbits==32) return ~0;
    assert(nbits < 32);
    return (1 << nbits) - 1;
}
static inline uint32_t bits_get(uint32_t x, unsigned lb, unsigned ub) {
    assert(lb <= ub && ub < 32);
    return (x >> lb) & bits_mask(ub-lb+1);
}
static inline uint32_t bits_set(uint32_t x, unsigned lb, unsigned ub, uint32_t v) {
    assert(lb <= ub && ub < 32);
    unsigned n = ub-lb+1;
    uint32_t mask = bits_mask(n);
    return (x & ~(mask << lb)) | (v << lb);
}

#endif
