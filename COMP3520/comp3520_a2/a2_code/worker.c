// worker.c  — run until parent kills us with SIGINT; SIGTSTP/SIGCONT handled by kernel.
// Build: gcc -O2 -Wall -Wextra -std=c11 -o worker worker.c
#define _XOPEN_SOURCE 700
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;
static void on_int(int sig){ (void)sig; running = 0; }

int main(int argc, char **argv){
    // print the id
    if(argc>1) fprintf(stderr, "[worker %s] started (pid=%d)\n", argv[1], getpid());

    struct sigaction sa = {0};
    sa.sa_handler = on_int;
    sigaction(SIGINT, &sa, NULL);

    // Busy loop to consume CPU; gets automatically stopped/continued by SIGTSTP/SIGCONT.
    // No exit until SIGINT from parent
    volatile unsigned long x = 0;
    while(running){
        // idk what to put here? 
        x += 1;
        if((x & 0xFFFFFF) == 0) {};
    }
    if(argc>1) fprintf(stderr, "[worker %s] exiting\n", argv[1]);
    return 0;
}

