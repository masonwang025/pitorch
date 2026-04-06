#include "deadbeef_shader.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
__declspec(align(8))
#elif defined(__GNUC__)
__attribute__((aligned(8)))
#endif
uint32_t deadbeef_shader[32] = {
0x00401a00, 0xe0021c67,
0xdeadbeef, 0xe0020c27,
0x159f2fc0, 0x100009e7,
0xbeefdead, 0xe0020c27,
0x159f2fc0, 0x100009e7,
0xfaded070, 0xe0020c27,
0x159f2fc0, 0x100009e7,
0xfeedface, 0xe0020c27,
0x159f2fc0, 0x100009e7,
0x82104000, 0xe0021c67,
0x15827d80, 0x10020827,
0x159e7000, 0x10021ca7,
0x159f2fc0, 0x100009e7,
0x009e7000, 0x300009e7,
0x009e7000, 0x100009e7,
0x009e7000, 0x100009e7
};
#ifdef __HIGHC__
#pragma Align_to(8, deadbeef_shader)
#ifdef __cplusplus
}
#endif
#endif
