CC=gcc
CFLAGS=-O2 -Wall -Wextra -std=c11

all: main worker

mlq: main.c worker.c
	$(CC) $(CFLAGS) -o main main.c
	$(CC) $(CFLAGS) -o worker worker.c

clean:
	rm -f main worker

