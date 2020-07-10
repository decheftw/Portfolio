# Makefile for the "AMStartup" program
#
# Devon Chen, Alex Feng, Abubakar Kasule, Katherine Taylor, February 2020
# CDds
 
L = ./libcs50

CC = gcc
CFLAGS = -Wall -pedantic -std=c11 -ggdb -pthread -I$L
PROGS = AMStartup #toolkit avatar AMStartup
OBJS = toolkit.o newavatar.o AMStartup.o
LIBS = -lm
LLIBS = $L/libcs50.a

.PHONY: all clean

all : $(PROGS)

# executable depends on object files
#toolkit: toolkit.o
#	$(CC) $(CFLAGS) $^ -o $@
#avatar: avatar.o
#	$(CC) $(CFLAGS) $^ -o $@
AMStartup: $(OBJS) $(LLIBS)
	$(CC) $(CFLAGS) $^ -o $@

# object files depend on include files
toolkit.o: toolkit.h
newavatar.o: toolkit.h amazing.h  
AMStartup.o: avatar.h toolkit.h amazing.h

test: $(PROGS) # testing.sh
	# bash -v testing.sh

clean:
	rm -f *~ *.o
	rm -rf *.dSYM
	rm -f core
	rm -f vgcore.*
	rm AMStartup
