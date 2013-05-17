#!/usr/bin/env python

import sys

if len(sys.argv)<2:
    print "usage: %s <code file>" % sys.argv[0]
    sys.exit(-1)

inputfile=sys.argv[1]

PATTERN = "{\n  __pyx_t_5numpy_uint16_t movieID;\n  __pyx_t_5numpy_uint32_t userID;\n  __pyx_t_5numpy_float32_t rating;\n}"
INSTRUMENTATION=" __attribute__((packed))"
f=open(inputfile,'r')
code=''.join(f.readlines())
f.close()
idx=code.find(PATTERN)

new_code=''.join([code[0:idx+len(PATTERN)],INSTRUMENTATION,code[idx+len(PATTERN):]])

g=open(inputfile,'w+')
g.write(new_code)
g.close()

