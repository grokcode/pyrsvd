
CC = gcc-4.8
LD = gcc-4.8
CYTHON = cython

NUMPY_PATH=/Users/takashi/.pyenv/versions/mac-2.7.9/lib/python2.7/site-packages/numpy/core/include
#gcc -bundle -undefined dynamic_lookup -L/usr/local/opt/readline/lib -L/usr/local/opt/readline/lib -L/Users/takashi/.pyenv/versions/2.7.9/lib -L/usr/local/opt/openssl/lib -L/usr/local/opt/openssl/lib -I/usr/local/opt/openssl/include build/temp.macosx-10.9-x86_64-2.7/rsvd/rsvd.o -o /Users/takashi/Downloads/pyrsvd/rsvd/rsvd.so -O3 -ffast-math
CC_FLAGS = -g -bundle -undefined dynamic_lookup -O3 -Wall -fno-strict-aliasing -I$(NUMPY_PATH) -I/Users/takashi/.pyenv/versions/mac-2.7.9/include/python2.7
LD_FLAGS = -g -o

MODULE_SO = rsvd/rsvd.so
ARCHIVE = pyrsvd.tar

$(MODULE_SO) : rsvd/rsvd.c
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(MODULE_SO) rsvd/rsvd.c

rsvd/rsvd.c : rsvd/rsvd.pyx
	$(CYTHON) rsvd/rsvd.pyx
	./instrument.py rsvd/rsvd.c

all : $(MODULE_SO)

clean : 
	rm $(MODULE_SO)
	rm rsvd/*.pyc

cleancython : 
	rm rsvd/rsvd.c

tar : 
	tar -cf $(ARCHIVE) --exclude-vcs --exclude models --exclude data --exclude build --exclude doc --exclude *~ --exclude svn-commit.tmp --exclude *.pyc --exclude *.so *
