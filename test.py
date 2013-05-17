#!/usr/bin/python

import sys
import time
print sys.argv[1]
f = open(sys.argv[1],"r")


for line in f:
    print line
    time.sleep(1)

f.close()
