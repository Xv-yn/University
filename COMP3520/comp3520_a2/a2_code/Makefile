CC=gcc
CFLAGS=-O2 -Wall -Wextra -std=c11

all: main

sim: simulation.c
	$(CC) $(CFLAGS) -o sim simulation.c

main: main.c
	$(CC) $(CFLAGS) -o main main.c

sig: sigtrap.c
	$(CC) $(CFLAGS) -o process sigtrap.c

clean:
	rm -f main process sim

