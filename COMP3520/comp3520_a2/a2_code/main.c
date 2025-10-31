
// main.c  — COMP3520 A2: Multi-Level Queue Dispatcher (real processes)
// Build:   gcc -O2 -Wall -Wextra -std=c11 -o mlq main.c
// Run:   ./main        (prompts, then paste jobs; Ctrl-D)

#define _XOPEN_SOURCE 700
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <errno.h>

typedef struct Job {
    int    id;
    int    arrival;
    int    burst_total;        // CPU time in seconds
    int    burst_left;
    int    init_prio;          // 0,1,2
    int    level;              // 0,1,2 when enqueued; -1 otherwise
    int    ran_in_level;       // accumulated time at current level
    int    first_start;        // time first got CPU (sim time)
    int    finish_time;        // time completed (sim time)
    int    wait_total;         // total waiting time (sim time)
    int    wait_since_tail;    // for W logic
    pid_t  pid;                // child PID (worker)
    bool   started;
    struct Job *next;
} Job;

typedef struct { Job *head, *tail; int size; } Queue;
static void q_init(Queue *q){ q->head=q->tail=NULL; q->size=0; }
static bool q_empty(Queue *q){ return q->size==0; }
static void q_push_tail(Queue *q, Job *j){
    j->next=NULL;
    if(q->tail) q->tail->next=j; else q->head=j;
    q->tail=j; q->size++;
}
static void q_push_head(Queue *q, Job *j){
    j->next=q->head; q->head=j; if(!q->tail) q->tail=j; q->size++;
}
static Job* q_pop_head(Queue *q){
    if(!q->head) return NULL;
    Job *j=q->head; q->head=j->next; if(!q->head) q->tail=NULL; j->next=NULL; q->size--; return j;
}

typedef struct { Job **a; int n, cap; } Vec;
static void v_init(Vec *v){ v->n=0; v->cap=16; v->a=malloc(sizeof(Job*)*v->cap); }
static void v_push(Vec *v, Job* j){
    if(v->n==v->cap){ v->cap*=2; v->a=realloc(v->a,sizeof(Job*)*v->cap); }
    v->a[v->n++]=j;
}
static int cmp_arrival(const void *pa, const void *pb){
    const Job *A = *(Job**)pa, *B = *(Job**)pb;
    if(A->arrival != B->arrival) return A->arrival - B->arrival;
    return A->id - B->id;
}

static int parse_job_line(const char *line, int *A,int *B,int *P){
    char buf[256]; size_t n=strlen(line); if(n>=sizeof(buf)) n=sizeof(buf)-1;
    memcpy(buf,line,n); buf[n]='\0';
    for(size_t i=0;i<n;i++) if(buf[i]==',') buf[i]=' ';
    for(char *p=buf; *p; ++p){ if(*p=='#'){ *p='\0'; break; } }
    int a,b,p; char *s=buf; while(*s==' '||*s=='\t') s++;
    if(*s=='\0' || *s=='\n') return 0;
    if(sscanf(s,"%d %d %d",&a,&b,&p)==3){ *A=a; *B=b; *P=p; return 1; }
    return 0;
}

static void read_jobs(FILE *in, Vec *jobs){
    v_init(jobs);
    char line[256]; int id=0, A,B,P;
    while(fgets(line,sizeof(line),in)){
        if(parse_job_line(line,&A,&B,&P)){
            Job *j = calloc(1,sizeof(Job));
            j->id=id++; j->arrival=A; j->burst_total=B; j->burst_left=B;
            j->init_prio=P; j->level=-1; j->ran_in_level=0;
            j->first_start=-1; j->finish_time=-1; j->wait_total=0; j->wait_since_tail=0;
            j->pid=-1; j->started=false; j->next=NULL;
            v_push(jobs,j);
        }
    }
    if(jobs->n==0){ fprintf(stderr,"No jobs parsed.\n"); exit(1); }
}

// Aging rules
static void move_all_preserve(Queue *from, Queue *to, int new_level){
    while(!q_empty(from)){
        Job *j = q_pop_head(from);
        j->level = new_level;
        j->ran_in_level = 0;
        j->wait_since_tail = 0;
        q_push_tail(to, j);
    }
}
static void starvation_boost(Queue *L0, Queue *L1, Queue *L2, int W){
    if(!q_empty(L1) && L1->head->wait_since_tail >= W){
        move_all_preserve(L1, L0, 0);
        move_all_preserve(L2, L0, 0);
        return;
    }
    if(!q_empty(L2) && L2->head->wait_since_tail >= W){
        move_all_preserve(L2, L0, 0);
    }
}

static void incr_waits_queue(Queue *q){
    for(Job *p=q->head;p;p=p->next){ p->wait_total++; p->wait_since_tail++; }
}
static void incr_waits_all(Queue *l0, Queue *l1, Queue *l2){
    incr_waits_queue(l0); incr_waits_queue(l1); incr_waits_queue(l2);
}

// Wall-clock sleep for one "tick" (1 second)
static void sleep_one_tick(void){
    struct timespec req = { .tv_sec = 1, .tv_nsec = 0 };
    while(nanosleep(&req, &req) == -1 && errno == EINTR) { /* retry */ }
}

// Start worker ./worker <id> (child runs an infinite CPU loop; terminate with SIGINT)
static void ensure_started(Job *j){
    if(j->started) return;
    pid_t pid = fork();
    if(pid < 0){ perror("fork"); exit(1); }
    if(pid == 0){
        char idbuf[32];
        snprintf(idbuf,sizeof(idbuf),"%d", j->id);
        char *argv[] = { "./process", idbuf, NULL };
        execv("./process", argv);
        perror("execv ./process");
        _exit(127);
    }
    j->pid = pid;
    j->started = true;
}

// Send a signal and synchronize if needed
static void send_and_sync(pid_t pid, int sig, int options){
    if(pid <= 0) return;
    if(kill(pid, sig) == -1 && errno != ESRCH){ perror("kill"); }
    if(options){
        // Wait for stop or exit, depends on options
        int status;
        if(waitpid(pid, &status, options) == -1 && errno != ECHILD){
            perror("waitpid");
        }
    }
}

int main(int argc, char **argv){
    int t0=-1,t1=-1,t2=-1,W=-1;
    FILE *jin = NULL;

    if(argc==5 || argc==6){
        t0=atoi(argv[1]); t1=atoi(argv[2]); t2=atoi(argv[3]); W=atoi(argv[4]);
        jin = (argc==6) ? fopen(argv[5],"r") : stdin;
        if(!jin){ perror("open jobs"); return 1; }
    }else{
        printf("Enter t0 t1 t2 W: ");
        fflush(stdout);
        if(scanf("%d %d %d %d",&t0,&t1,&t2,&W)!=4){ fprintf(stderr,"Failed to read params.\n"); return 1; }
        int c; while((c=getchar())!='\n' && c!=EOF);
        printf("Paste jobs (arrival cputime priority), commas optional. Ctrl-D when done:\n");
        jin = stdin;
    }
    if(t0<=0||t1<=0||t2<=0||W<0){ fprintf(stderr,"Bad params.\n"); return 1; }

    Vec jobs; read_jobs(jin,&jobs);
    if(jin!=stdin) fclose(jin);
    qsort(jobs.a, jobs.n, sizeof(Job*), cmp_arrival);

    Queue L0,L1,L2; q_init(&L0); q_init(&L1); q_init(&L2);
    int now=0, next_i=0, finished=0;
    Job *running=NULL;

    // Ensure reap any children that exit
    struct sigaction sa = {0};
    sa.sa_handler = SIG_IGN;   // children are waited explicitly
    sigaction(SIGPIPE, &sa, NULL);

    while(finished < jobs.n){
        // Load arrivals at 'now' into their initial queues
        bool saw_lvl0=false, saw_lvl1=false;
        while(next_i<jobs.n && jobs.a[next_i]->arrival==now){
            Job *j = jobs.a[next_i++];
            if(j->init_prio<=0){ j->level=0; j->ran_in_level=0; j->wait_since_tail=0; q_push_tail(&L0,j); saw_lvl0=true; }
            else if(j->init_prio==1){ j->level=1; j->ran_in_level=0; j->wait_since_tail=0; q_push_tail(&L1,j); saw_lvl1=true; }
            else { j->level=2; j->ran_in_level=0; j->wait_since_tail=0; q_push_tail(&L2,j); }
        }

        // Immediate preemptions on arrival
        if(running){
            if(running->level==2 && (saw_lvl0 || saw_lvl1)){
                // L2 preempt on L0/L1 arrival -> to HEAD of L2
                send_and_sync(running->pid, SIGTSTP, WUNTRACED);
                q_push_head(&L2, running);
                running=NULL;
            }else if(running->level==1 && saw_lvl0){
                // L1 preempt on L0 arrival -> to HEAD of L1
                send_and_sync(running->pid, SIGTSTP, WUNTRACED);
                q_push_head(&L1, running);
                running=NULL;
            }
        }

        // Aging before dispatch
        starvation_boost(&L0,&L1,&L2,W);

        // If CPU idle, pick next
        if(!running){
            if(!q_empty(&L0))      running = q_pop_head(&L0);
            else if(!q_empty(&L1)) running = q_pop_head(&L1);
            else if(!q_empty(&L2)) running = q_pop_head(&L2);
        }

        // If still no work, fast-forward time to next arrival
        if(!running){
            if(next_i<jobs.n){ now = jobs.a[next_i]->arrival; continue; }
            break;
        }

        // Ensure child exists, then (re)start/resume it
        ensure_started(running);
        if(running->first_start<0) running->first_start = now;
        // Continue it if it’s stopped
        send_and_sync(running->pid, SIGCONT, 0);

        // Decide how much to let it run for this tick (1s granularity)
        // Simulate time at 1-second ticks to keep accounting consistent with spec
        incr_waits_all(&L0,&L1,&L2);      // everyone else waits during this 1s
        sleep_one_tick();                  // wall clock 1s

        // After 1s, account
        running->burst_left--;
        running->ran_in_level++;
        now++;

        // If child finish on own, detect via waitpid
        // Terminate explicitly when burst_left reaches zero
        if(running->burst_left==0){
            send_and_sync(running->pid, SIGINT, 0);                 // terminate
            int status; waitpid(running->pid, &status, 0);          // reap
            running->finish_time = now;
            running = NULL; finished++;
            continue;
        }

        // Check if quantum for the current level has been consumed
        if(running->level==0 && running->ran_in_level>=t0){
            send_and_sync(running->pid, SIGTSTP, WUNTRACED);
            running->level = 1;
            running->ran_in_level = 0;
            running->wait_since_tail = 0;
            q_push_tail(&L1, running);
            running=NULL;
        }else if(running->level==1 && running->ran_in_level>=t1){
            send_and_sync(running->pid, SIGTSTP, WUNTRACED);
            running->level = 2;
            running->ran_in_level = 0;
            running->wait_since_tail = 0;
            q_push_tail(&L2, running);
            running=NULL;
        }else if(running->level==2 && running->ran_in_level>=t2){
            send_and_sync(running->pid, SIGTSTP, WUNTRACED);
            running->ran_in_level = 0;
            running->wait_since_tail = 0;
            q_push_tail(&L2, running);
            running=NULL;
        }
    }

    // Termination guard (checks if anything leaked)
    for(int i=0;i<jobs.n;i++){
        if(jobs.a[i]->started){
            kill(jobs.a[i]->pid, SIGINT);
            waitpid(jobs.a[i]->pid, NULL, 0);
        }
    }

    // Metrics
    double sumT=0,sumW=0,sumR=0;
    for(int i=0;i<jobs.n;i++){
        Job *j=jobs.a[i];
        int T = j->finish_time - j->arrival;
        int R = j->first_start - j->arrival;
        sumT+=T; sumW+=j->wait_total; sumR+=R;
    }
    printf("Average Turnaround Time: %.3f\n", sumT/jobs.n);
    printf("Average Waiting Time:    %.3f\n", sumW/jobs.n);
    printf("Average Response Time:   %.3f\n", sumR/jobs.n);

    for(int i=0;i<jobs.n;i++) free(jobs.a[i]);
    free(jobs.a);
    return 0;
}
