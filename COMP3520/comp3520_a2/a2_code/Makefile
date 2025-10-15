CC=gcc
CFLAGS=-O2 -Wall -Wextra -std=c11

all: main worker

sim: simulation.c
	$(CC) $(CFLAGS) -o sim simulation.c

main: main.c sigtrap.c
	$(CC) $(CFLAGS) -o main main.c
	$(CC) $(CFLAGS) -o process sigtrap.c


clean:
	rm -f main process sim

