
CC = gcc-4.8
LD = gcc-4.8
CYTHON = cython

NUMPY_PATH=/Users/takashi/.pyenv/versions/mac-2.7.9/lib/python2.7/site-packages/numpy/core/include

CC_FLAGS = -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I$(NUMPY_PATH) -I/usr/include/python2.7
LD_FLAGS = -o

MODULE_SO = rsvd/rsvd.so
ARCHIVE = pyrsvd.tar

rsvd.so : rsvd.c
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(MODULE_SO) rsvd/rsvd.c

rsvd.c : rsvd/rsvd.pyx
	$(CYTHON) rsvd/rsvd.pyx
	./instrument.py rsvd/rsvd.c

all : rsvd.so

clean : 
	rm $(MODULE_SO)
	rm rsvd/*.pyc

cleancython : 
	rm rsvd/rsvd.c

tar : 
	tar -cf $(ARCHIVE) --exclude-vcs --exclude models --exclude data --exclude build --exclude doc --exclude *~ --exclude svn-commit.tmp --exclude *.pyc --exclude *.so *
