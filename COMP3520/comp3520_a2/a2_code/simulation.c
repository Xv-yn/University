#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Usage:
   1) ./sim t0 t1 t2 W < jobs.txt          # read jobs from stdin
   2) ./sim t0 t1 t2 W jobs.txt            # read jobs from file
   3) ./sim                                 # interactive: enter t0 t1 t2 W, then paste jobs, Ctrl-D
   Job lines may be "A B P" or "A, B, P" (commas optional)
*/

typedef struct Job {
    int id;
    int arrival;
    int burst_total;
    int burst_left;
    int init_prio;             // initial priority (0..2)
    int level;                 // current level (0..2), -1 if not in a ready queue yet

    int ran_in_level;          // accumulated CPU time at current level since last TAIL placement

    int first_start;           // first time it got CPU (-1 until it runs)
    int finish_time;           // completion time
    int wait_total;            // total waiting time (all queues)
    int wait_since_tail;       // waiting time SINCE last placement at the TAIL of this level (for W checks)

    struct Job *next;
} Job;

/* Simple queue */
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
static void q_move_all_preserve_order(Queue *from, Queue *to){
    if(from->size==0) return;
    if(to->tail) to->tail->next = from->head; else to->head = from->head;
    to->tail = from->tail;
    to->size += from->size;
    from->head = from->tail = NULL; from->size = 0;
}

/* dyn array of pointers */
typedef struct { Job **a; int n, cap; } Vec;
static void v_init(Vec *v){ v->n=0; v->cap=16; v->a=(Job**)malloc(sizeof(Job*)*v->cap); }
static void v_push(Vec *v, Job* j){
    if(v->n==v->cap){ v->cap*=2; v->a=(Job**)realloc(v->a,sizeof(Job*)*v->cap); }
    v->a[v->n++]=j;
}
static int cmp_arrival(const void *pa, const void *pb){
    Job *A=*(Job**)pa, *B=*(Job**)pb;
    if(A->arrival!=B->arrival) return A->arrival-B->arrival;
    return A->id-B->id;
}

/* bookkeeping helpers */
static void incr_waits_queue(Queue *q){
    for(Job *p=q->head;p;p=p->next){ p->wait_total++; p->wait_since_tail++; }
}
static void incr_waits_all(Queue *l0, Queue *l1, Queue *l2){
    incr_waits_queue(l0); incr_waits_queue(l1); incr_waits_queue(l2);
}

/* enqueue at TAIL: set level, reset ran_in_level, reset wait_since_tail */
static void enqueue_tail(Queue *q, Job *j, int level){
    j->level = level;
    j->ran_in_level = 0;
    j->wait_since_tail = 0;
    q_push_tail(q, j);
}
/* enqueue at HEAD due to preemption; do NOT reset ran_in_level or wait_since_tail */
static void enqueue_head(Queue *q, Job *j, int level){
    j->level = level;
    q_push_head(q, j);
}

/* starvation rules (W):
   - If HEAD of L1 has wait_since_tail >= W: move ALL L1, then ALL L2 to END of L0 (preserve order)
   - Else if HEAD of L2 has wait_since_tail >= W: move ALL L2 to END of L0
*/
static void starvation_boost(Queue *L0, Queue *L1, Queue *L2, int W){
    if(!q_empty(L1) && L1->head->wait_since_tail >= W){
        Queue tmp; q_init(&tmp);
        while(!q_empty(L1)){ enqueue_tail(&tmp, q_pop_head(L1), 0); }
        while(!q_empty(L2)){ enqueue_tail(&tmp, q_pop_head(L2), 0); }
        q_move_all_preserve_order(&tmp, L0);
        return;
    }
    if(!q_empty(L2) && L2->head->wait_since_tail >= W){
        Queue tmp; q_init(&tmp);
        while(!q_empty(L2)){ enqueue_tail(&tmp, q_pop_head(L2), 0); }
        q_move_all_preserve_order(&tmp, L0);
    }
}

/* parse a job line; accepts commas or spaces; returns 1 if parsed, 0 otherwise */
static int parse_job_line(const char *line, int *A,int *B,int *P){
    // copy and replace commas with spaces
    char buf[256]; size_t n=strlen(line);
    if(n>=sizeof(buf)) n=sizeof(buf)-1;
    memcpy(buf,line,n); buf[n]='\0';
    for(size_t i=0;i<n;i++) if(buf[i]==',') buf[i]=' ';
    // skip comments
    for(char *p=buf; *p; ++p){ if(*p=='#'){ *p='\0'; break; } }
    int a,b,p;
    char *s=buf;
    // consume leading spaces
    while(*s==' '||*s=='\t') s++;
    if(*s=='\0' || *s=='\n') return 0;
    if(sscanf(s,"%d %d %d",&a,&b,&p)==3){
        *A=a; *B=b; *P=p; return 1;
    }
    return 0;
}

static void read_jobs(FILE *in, Vec *jobs){
    v_init(jobs);
    char line[256];
    int id=0, A,B,P;
    while(fgets(line,sizeof(line),in)){
        if(parse_job_line(line,&A,&B,&P)){
            Job *j=(Job*)calloc(1,sizeof(Job));
            j->id=id++; j->arrival=A; j->burst_total=j->burst_left=B;
            j->init_prio=P; j->level=-1; j->ran_in_level=0;
            j->first_start=-1; j->finish_time=-1;
            j->wait_total=0; j->wait_since_tail=0; j->next=NULL;
            v_push(jobs,j);
        }
    }
    if(jobs->n==0){
        fprintf(stderr,"No jobs parsed. Provide lines like: 0 3 0  or  0, 3, 0\n");
        exit(1);
    }
}

int main(int argc, char **argv){
    int t0=-1,t1=-1,t2=-1,W=-1;
    FILE *jin = NULL;

    if(argc==5 || argc==6){
        t0=atoi(argv[1]); t1=atoi(argv[2]); t2=atoi(argv[3]); W=atoi(argv[4]);
        if(argc==6){
            jin=fopen(argv[5],"r");
            if(!jin){ perror("open jobs file"); return 1; }
        }else{
            jin=stdin;
        }
    }else{
        // interactive: ask times first, then read jobs from stdin until EOF
        printf("Enter t0 t1 t2 W: ");
        fflush(stdout);
        if(scanf("%d %d %d %d",&t0,&t1,&t2,&W)!=4){
            fprintf(stderr,"Failed to read t0 t1 t2 W.\n"); return 1;
        }
        // consume the newline after the W so fgets works cleanly
        int c; while((c=getchar())!='\n' && c!=EOF);
        printf("Paste jobs (arrival cputime priority), commas optional. Ctrl-D when done:\n");
        jin=stdin;
    }
    if(t0<=0||t1<=0||t2<=0||W<0){
        fprintf(stderr,"Invalid parameters: t0=%d t1=%d t2=%d W=%d\n",t0,t1,t2,W);
        return 1;
    }

    Vec jobs; read_jobs(jin,&jobs);
    if(jin!=stdin) fclose(jin);
    qsort(jobs.a, jobs.n, sizeof(Job*), cmp_arrival);

    Queue L0,L1,L2; q_init(&L0); q_init(&L1); q_init(&L2);
    int now=0, next_i=0, finished=0;
    Job *running=NULL;

    // main simulation loop
    while(finished < jobs.n){
        // bring in arrivals at 'now'
        bool saw_lvl0=false, saw_lvl1=false;
        while(next_i<jobs.n && jobs.a[next_i]->arrival==now){
            Job *j=jobs.a[next_i++];
            int lvl = j->init_prio; if(lvl<0) lvl=0; if(lvl>2) lvl=2;
            if(lvl==0){ enqueue_tail(&L0,j,0); saw_lvl0=true; }
            else if(lvl==1){ enqueue_tail(&L1,j,1); saw_lvl1=true; }
            else { enqueue_tail(&L2,j,2); }
        }

        // preemptions triggered by arrivals (immediate)
        if(running){
            if(running->level==2 && (saw_lvl0 || saw_lvl1)){
                enqueue_head(&L2, running, 2); running=NULL;
            }else if(running->level==1 && saw_lvl0){
                enqueue_head(&L1, running, 1); running=NULL;
            }
        }

        // starvation prevention before choosing next
        starvation_boost(&L0,&L1,&L2,W);

        // dispatch if CPU idle
        if(!running){
            if(!q_empty(&L0)){ running=q_pop_head(&L0); }
            else if(!q_empty(&L1)){ running=q_pop_head(&L1); }
            else if(!q_empty(&L2)){ running=q_pop_head(&L2); }
        }

        // if still no work, fast-forward to next arrival
        if(!running){
            if(next_i<jobs.n){ now = jobs.a[next_i]->arrival; continue; }
            break; // should not happen unless done
        }

        // mark first start
        if(running->first_start<0) running->first_start = now;

        // run for one time unit
        incr_waits_all(&L0,&L1,&L2);          // others wait during this tick
        running->burst_left--;
        running->ran_in_level++;
        now++;

        // finished?
        if(running->burst_left==0){
            running->finish_time = now;
            running=NULL; finished++;
            continue;
        }

        // time-quantum handling per level (based on ACCUMULATED run at this level)
        if(running){
            if(running->level==0 && running->ran_in_level>=t0){
                // demote to L1 tail
                enqueue_tail(&L1, running, 1);
                running=NULL;
            }else if(running->level==1 && running->ran_in_level>=t1){
                // demote to L2 tail
                enqueue_tail(&L2, running, 2);
                running=NULL;
            }else if(running->level==2 && running->ran_in_level>=t2){
                // RR in L2: place at END of L2
                enqueue_tail(&L2, running, 2);
                running=NULL;
            }
        }
    }

    // metrics
    double sum_turn=0, sum_wait=0, sum_resp=0;
    for(int i=0;i<jobs.n;i++){
        Job *j=jobs.a[i];
        int turnaround = j->finish_time - j->arrival;
        int response   = j->first_start - j->arrival;
        sum_turn += turnaround;
        sum_wait += j->wait_total;
        sum_resp += response;
    }
    printf("Average Turnaround Time: %.3f\n", sum_turn / jobs.n);
    printf("Average Waiting Time:    %.3f\n", sum_wait / jobs.n);
    printf("Average Response Time:   %.3f\n", sum_resp / jobs.n);

    for(int i=0;i<jobs.n;i++) free(jobs.a[i]);
    free(jobs.a);
    return 0;
}
