#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>

typedef struct {
    int size;
    int room_assigned;   // -1 until teacher assigns
    int entered_cnt;
    int left_cnt;
    bool called;
    bool completed;
    
    pthread_cond_t called_cv;
    pthread_cond_t all_entered_cv;
    pthread_cond_t all_left_cv;
    pthread_cond_t completed_cv;
} Group;

typedef struct {
    int id;
    int current_group;        // -1 when empty
    pthread_cond_t assigned_cv;
} Room;

// ----------- globals (protected by mu) -----------
int N, M, K, Tlim;
int *group_of_student;        // size N, set by teacher
Group *G;
Room *R;

pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t all_arrived_cv = PTHREAD_COND_INITIALIZER;
pthread_cond_t empty_room_cv  = PTHREAD_COND_INITIALIZER;
pthread_cond_t assigned_one_cv = PTHREAD_COND_INITIALIZER; // teacher -> specific student
pthread_cond_t ack_cv          = PTHREAD_COND_INITIALIZER; // student -> teacher
pthread_cond_t rooms_phase_cv = PTHREAD_COND_INITIALIZER;

int arrived_cnt = 0;
int next_gid = 0;
int groups_finished = 0;
bool all_groups_assigned = false;
bool all_groups_done = false;
int next_sid = -1;
int ack_sid  = -1;
bool rooms_phase_started = false;

// naive queue for empty rooms
int *empty_rooms; int er_head=0, er_tail=0, er_cap=0;
static void er_push(int x){ empty_rooms[er_tail++]=x; }
static bool er_empty(){ return er_head==er_tail; }
static int er_pop(){ return empty_rooms[er_head++]; }

static void print_assign_groups_start(){
    printf("Teacher: All students have arrived. I start to assign group ids to students.\n"); fflush(stdout);
}
static void print_student_arrived(int id){
    printf("Student [%d]: I have arrived and wait for being assigned to a group.\n", id); fflush(stdout);
}
static void print_assigned_student(int sid, int gid){
    printf("Teacher: student [%d] is in group [%d].\n", sid, gid); fflush(stdout);
}
static void print_student_knows_group(int sid, int gid){
    printf("Student [%d]: OK, I'm in group [%d] and waiting for my turn to enter a lab room.\n", sid, gid); fflush(stdout);
}
static void print_teacher_waiting_rooms(){
    printf("Teacher: I'm waiting for lab rooms to become available.\n"); fflush(stdout);
}
static void print_teacher_room_available(int rid){
    printf("Teacher: The lab [%d] is now available. Students in group [%d] can enter the room and start your lab exercise.\n",
           rid, R[rid].current_group); fflush(stdout);
}
static void print_tutor_room_ready(int rid){
    printf("Tutor [%d]: The lab room [%d] is vacated and ready for one group.\n", rid, rid); fflush(stdout);
}
static void print_student_entering(int sid, int gid, int rid){
    printf("Student [%d] in group [%d]: My group is called. I will enter the lab room [%d] now.\n", sid, gid, rid); fflush(stdout);
}
static void print_tutor_all_in(int rid, int gid){
    printf("Tutor [%d]: All students in group [%d] have entered the room [%d]. You can start your exercise now.\n", rid, gid, rid); fflush(stdout);
}
static void print_tutor_done(int rid, int gid, int took){
    printf("Tutor [%d]: Students in group[%d] have completed the lab exercise in [%d] units of time. You may leave this room now.\n", rid, gid, took); fflush(stdout);
}
static void print_student_bye(int sid, int gid){
    printf("Student [%d] in group [%d]: Thanks Tutor! Bye!\n", sid, gid); fflush(stdout);
}
static void print_teacher_no_students_tutor_go(int rid){
    printf("Teacher: There are no students waiting. Tutor [%d], you can go home now.\n", rid); fflush(stdout);
}
static void print_tutor_bye(int rid){
    printf("Tutor [%d]: Thanks Teacher. Bye!\n", rid); fflush(stdout);
}
static void print_teacher_all_left(){
    printf("Teacher: All students and tutors are left. I can now go home.\n"); fflush(stdout);
}

void *student_thread(void *arg){
    int sid = (int)(long)arg;

    pthread_mutex_lock(&mu);
    print_student_arrived(sid);
    if (++arrived_cnt == N) pthread_cond_broadcast(&all_arrived_cv);

    // Wait by sleeping on all_arrived_cv until teacher announces assigned
    while (group_of_student[sid] == -1 || next_sid != sid) pthread_cond_wait(&assigned_one_cv, &mu);
    int gid = group_of_student[sid];
    print_student_knows_group(sid, gid);

    ack_sid = sid;
    pthread_cond_signal(&ack_cv);
    
    // wait until teacher calls this group (room assigned)
    while (!G[gid].called) pthread_cond_wait(&G[gid].called_cv, &mu);
    int rid = G[gid].room_assigned;
    print_student_entering(sid, gid, rid);

    // enter: increment entered count; last signals tutor
    if (++G[gid].entered_cnt == G[gid].size) pthread_cond_signal(&G[gid].all_entered_cv);

    // Wait for tutor to announce completion before saying bye
    while (!G[gid].completed)
        pthread_cond_wait(&G[gid].completed_cv, &mu);

    // Leave: print bye, increment left; if last, signal tutor
    print_student_bye(sid, gid);

    if (++G[gid].left_cnt == G[gid].size)
        pthread_cond_broadcast(&G[gid].all_left_cv);    
    pthread_mutex_unlock(&mu);
    return NULL;
}

void *tutor_thread(void *arg){
    int rid = (int)(long)arg;
    pthread_mutex_lock(&mu);

    // Wait for room phase to begin
    while (!rooms_phase_started)
        pthread_cond_wait(&rooms_phase_cv, &mu);

    // Announce room initially empty
    R[rid].current_group = -1;
    print_tutor_room_ready(rid);
    er_push(rid);
    pthread_cond_signal(&empty_room_cv);

    while (!all_groups_done){
        // Wait until a group is assigned to this room
        while (R[rid].current_group == -1 && !all_groups_done)
            pthread_cond_wait(&R[rid].assigned_cv, &mu);
        if (all_groups_done) break;

        int gid = R[rid].current_group;

        // Wait for all students of gid to enter
        while (G[gid].entered_cnt < G[gid].size)
            pthread_cond_wait(&G[gid].all_entered_cv, &mu);
        print_tutor_all_in(rid, gid);

        // Simulate exercise outside the lock
        pthread_mutex_unlock(&mu);
        // Make duration strictly positive: 1..Tlim
        int duration = (Tlim <= 0) ? 1 : (rand() % Tlim) + 1;
        sleep(duration);
        pthread_mutex_lock(&mu);

        // Announce completion, THEN allow students to leave
        print_tutor_done(rid, gid, duration);
        G[gid].completed = true;
        pthread_cond_broadcast(&G[gid].completed_cv);

        // Wait for all to leave
        while (G[gid].left_cnt < G[gid].size)
            pthread_cond_wait(&G[gid].all_left_cv, &mu);

        // Room vacated; notify teacher
        G[gid].called = false; // finished
        R[rid].current_group = -1;
        print_tutor_room_ready(rid);
        er_push(rid);
        pthread_cond_signal(&empty_room_cv);

        // Track progress; if all groups done, let all tutors exit cleanly
        if (++groups_finished == M) {
            all_groups_done = true;
            // Wake any tutors waiting for an assignment so they can exit
            for (int rr = 0; rr < K; ++rr)
                pthread_cond_broadcast(&R[rr].assigned_cv);
        }
    }

    print_tutor_bye(rid);
    pthread_mutex_unlock(&mu);
    return NULL;
}

void *teacher_thread(void *arg){
    (void)arg;
    pthread_mutex_lock(&mu);

    printf("Teacher: I'm waiting for all students to arrive.\n");

    // wait all arrived
    while (arrived_cnt < N) pthread_cond_wait(&all_arrived_cv, &mu);

    // assign students to groups randomly as specified
    print_assign_groups_start();

    // build group sizes per spec
    int base = N / M, r = N % M;
    for (int g = 0; g < M; ++g) G[g].size = base + (g < r ? 1 : 0);
    // produce a random permutation of students and fill groups 0..M-1 in order
    int *perm = malloc(N*sizeof(int));
    for (int i=0;i<N;i++) perm[i]=i;
    for (int i=N-1;i>0;i--){ int j=rand()%(i+1); int t=perm[i]; perm[i]=perm[j]; perm[j]=t; }
    int p = 0;
    for (int g = 0; g < M; ++g) {
        for (int k = 0; k < G[g].size; ++k) {
            int sid = perm[p++];

            group_of_student[sid] = g;
            print_assigned_student(sid, g);

            // let only this student proceed
            next_sid = sid;
            pthread_cond_broadcast(&assigned_one_cv);

            // wait for this student to acknowledge
            while (ack_sid != sid)
                pthread_cond_wait(&ack_cv, &mu);
        }
    }

    rooms_phase_started = true;
    pthread_cond_broadcast(&rooms_phase_cv);
    print_teacher_waiting_rooms();

    // assign rooms to groups in natural order
    while (next_gid < M){
        while (er_empty()) pthread_cond_wait(&empty_room_cv, &mu);
        int rid = er_pop();

        // assign pair
        int gid = next_gid++;
        G[gid].called = true;
        G[gid].room_assigned = rid;
        R[rid].current_group = gid;
        print_teacher_room_available(rid);
        pthread_cond_broadcast(&G[gid].called_cv);
        pthread_cond_signal(&R[rid].assigned_cv);
    }
    int done_groups = 0;
    for (int g=0; g<M; ++g) while (G[g].left_cnt < G[g].size) pthread_cond_wait(&G[g].all_left_cv, &mu), (void)g;
    all_groups_done = true;
    // wake all idle tutors so they can exit
    for (int rId=0; rId<K; ++rId) pthread_cond_broadcast(&R[rId].assigned_cv);
    
    print_teacher_all_left();
    pthread_mutex_unlock(&mu);
    return NULL;
}

int main(){
    srand((unsigned)time(NULL));
    if (scanf("%d %d %d %d", &N, &M, &K, &Tlim) != 4) {
      fprintf(stderr, "Error: expected 4 integers: N M K Tlim.\n");
      return 1;
    }
    if (N <= 0 || M <= 0 || K <= 0 || Tlim <= 0) {
        fprintf(stderr, "Error: N, M, K, and Tlim must all be > 0. "
                    "Got N=%d M=%d K=%d Tlim=%d\n", N, M, K, Tlim);
        return 1;
    }
    // allocating memory
    group_of_student = malloc(N*sizeof(int));
    for (int i=0;i<N;i++) group_of_student[i] = -1;

    G = calloc(M, sizeof(Group));
    R = calloc(K, sizeof(Room));
    er_cap = K + M + 8;     // added buffer, just in case of overflow 
    empty_rooms = malloc(er_cap*sizeof(int));

    for (int g=0; g<M; ++g){
        G[g].room_assigned = -1;
        pthread_cond_init(&G[g].called_cv,NULL);
        pthread_cond_init(&G[g].all_entered_cv,NULL);
        pthread_cond_init(&G[g].all_left_cv,NULL);
        pthread_cond_init(&G[g].completed_cv, NULL);

    }
    for (int r=0; r<K; ++r){
        R[r].id = r;
        R[r].current_group = -1;
        pthread_cond_init(&R[r].assigned_cv,NULL);
    }

    // spawn threads
    pthread_t teacher, *tutors = malloc(K*sizeof(pthread_t)), *students = malloc(N*sizeof(pthread_t));
    pthread_create(&teacher, NULL, teacher_thread, NULL);
    for (int r=0; r<K; ++r) pthread_create(&tutors[r], NULL, tutor_thread, (void*)(long)r);
    for (int s=0; s<N; ++s) pthread_create(&students[s], NULL, student_thread, (void*)(long)s);

    // join
    for (int s=0; s<N; ++s) pthread_join(students[s], NULL);
    for (int r=0; r<K; ++r) pthread_join(tutors[r], NULL);
    pthread_join(teacher, NULL);

    printf("Main thread: All students have completed their lab exercises. This is the end of simulation.\n");
    return 0;
}
