#ifndef __DEMAND_H__
#define __DEMAND_H__
// Error-checking / debugging macros from cs140e.
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#define _XSTRING(x) #x

#define impossible(args...) do {                            \
    fprintf(stderr, "%s:%s:%d:IMPOSSIBLE ERROR:",           \
                        __FILE__, __FUNCTION__, __LINE__);  \
    fprintf(stderr, ##args);						        \
    exit(1);                                                \
} while(0)

#define output(msg...) fprintf(stderr, ##msg)

#define notreached()    panic("impossible: hit NOTREACHED!\n")
#define unimplemented() panic("this code is not implemented.\n")

#define AssertNow(x) switch(1) { case (x): case 0: ; }

#define trace(msg...) do { output("TRACE:"); output(msg); } while(0)

#ifdef NDEBUG
#   define demand(_expr, _msg, args...) do { } while(0)
#   define debug(msg...) do { } while(0)
#else
#   define demand(_expr, _msg, args...) do {			                \
      if(!(_expr)) { 							                        \
        fprintf(stderr, "%s:%s:%d: Assertion '%s' failed:",       \
                        __FILE__, __FUNCTION__, __LINE__, _XSTRING(_expr));           \
        fprintf(stderr, _XSTRING(_msg) "\n", ##args);	            \
        exit(1);						                            \
      }								                                \
    } while(0)

#   define debug_output(msg...) output("DEBUG:" msg)

#   define debug(msg...) do { 						                    \
        fprintf(stderr, "%s:%s:%d:", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, ##msg);						                    \
    } while(0)
#endif

#define sys_die(syscall, msg, args...) do {                                         \
    debug("FATAL syscall error: " _XSTRING(msg) "\n\tperror reason: ", ##args);     \
    perror(_XSTRING(syscall));                                                      \
    exit(1);                                                                        \
} while(0)

#define no_fail(syscall) do {                                               \
    if((syscall) < 0) {                                                      \
        int reason = errno;                                                 \
        sys_die(syscall,                                                    \
            "<%s> failed: errno=%d: we thought this was impossible.\n",     \
                            _XSTRING(syscall),reason);                      \
    }                                                                       \
} while(0)

#define clean_exit(msg...) do { 	\
    fprintf(stderr, ##msg);			\
    exit(0);                        \
} while(0)

#define die(msg...) do { 						                            \
    fprintf(stderr, ##msg);						                            \
    exit(1);                                                                \
} while(0)

#define panic(msg...) do { 						                            \
    output("%s:%s:%d:PANIC:", __FILE__, __FUNCTION__, __LINE__);   \
    die(msg); \
} while(0)

#define todo(msg) panic("TODO: %s\n", msg)

#define fatal(msg...) do { 						                            \
    output("%s:%s:%d:", __FILE__, __FUNCTION__, __LINE__);   \
    die(msg); \
} while(0)

#endif
