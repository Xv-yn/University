# Concurrency: Mutual Exclusion and Synchronization (cont.)

## Spinlocks and Sleeping locks

### Spinlocks

Sample Code:

```C
struct{
    int flag;           // 0 = unlocked, 1 = locked
} lock_t;

void init(lock_t *L) {
    L->flag = 0;        // set to unlocked
}

void lock(lock_t *L) {
    while (cmpxchg(&L->flag, 0, 1) == 1) { ; }
}

void unlock(lock_t *L) {
    L->flag = 0;        // set to unlocked
}
```

The key instruction: `cmpxchg(&L->flag, 0, 1)`

Meaning (compare-and-exchange): - “If *addr equals expected (here 0), atomically write new (here 1) to \*addr.
Return the old value that was in *addr.”

In the loop:

- `while (cmpxchg(&L->flag, 0, 1) == 1) { ; }`
- If flag was 1 (somebody else holds the lock), cmpxchg returns 1 and does not
  change the flag → the condition is true → keep spinning.
- If flag was 0 (free), cmpxchg atomically writes 1, returns 0, the condition is
  false → we exit the loop → lock acquired.

Why atomic matters: Two threads can’t both see 0 and both set it to 1—the CPU
guarantees only one wins.

#### Advantages

- Critical section is very short (a few dozen–few hundred cycles).
- Contention is low and the number of runnable threads ≲ number of cores.
- On a multicore CPU, a thread can spin on Core A while the lock holder runs on Core
  B and releases soon.

Analogy: If the bathroom trips are always <5 seconds and there are at most as many
people as bathrooms, waiting at the door is faster than walking away and coming back.

#### Disadvantages

- Busy-waiting wastes CPU cycles. On a single core, if the lock holder is preempted
  and a spinner is running, the holder can’t run to release → performance collapse.
  - All the people outside are just pacing and staring at the latch, not dancing, not
    getting snacks, not enjoying the concert. They’re burning energy doing nothing
    useful.
- Preemption/convoying: If the thread inside the critical section gets preempted,
  everyone else hammers the lock and burns CPU until the scheduler eventually runs the
  holder again.
  - Suppose the person inside is about to unlock, but suddenly gets a phone call
    (preempted by scheduler). Meanwhile, the entire line of people outside keeps
    rattling the latch every second, wasting time and crowding.
- No fairness guarantees:
  - A thread may spin forever under contention (livelock/starvation).
  - Spinlocks are usually unfair; recent arrivals can overtake.
    - When the stall opens, it’s not guaranteed that the first person in line goes in.
      Someone at the back may shove forward and grab it first.
- Not for long sections or blocking work: Don’t do I/O, syscalls that may sleep, or
  long loops while holding a spinlock.
  - Imagine someone goes into the stall not just to pee, but to take a shower, do
    their laundry, and make a phone call. Meanwhile everyone else is still staring at
    the latch, wasting energy.
- Priority inversion risk: a low-priority holder can block high-priority spinners
  indefinitely.
  - Suppose the person who grabbed the stall is a child taking forever (low-priority
    thread). Outside, the VIP artist (high-priority thread) is waiting in line. But the
    VIP still has to wait because the child won’t leave.

#### Realistic Usage

In a single-core vs multi-core situation:

Single-core user space: spinlocks are usually a bad idea (you spin instead of letting
the holder run).

Kernel/interrupt context: spinlocks are common because you often cannot sleep; also
you can disable preemption/interrupts around the critical section.

### Sleeping Lock

How Sleeping Locks Work

- Lock request:
  - If lock unavailable → suspend the thread, put it into a waiting queue.
- Unlock:
  - When the holder unlocks, it wakes up one waiting thread and moves it back to the
    ready queue.

Requires OS support → because putting threads to sleep and waking them up involves
scheduler + kernel.

#### futex

Futex = “Fast Userspace Mutex”

- Provided by Linux as a system call.
- Most of the time → threads do locking in userspace (fast).
- Only when there’s contention → call into the kernel (slow).
- Futex = integer in userspace + kernel wait queue.

Analogy:

- Like waiting at a café:
- If the table is free → just sit down (userspace, no kernel).
- If it’s busy → tell the waiter (kernel) to add you to the waitlist.

Futex operations:

- `futex_wait(addr, val)`
  - Put thread to sleep if `*addr == val`.
  - Returns immediately if value is different (someone already unlocked).

`futex_wake(addr, n)` - Wake up up to n threads waiting on addr.

Analogy:

- `futex_wait` = “If the light is red, go sit in the waiting area.”
- `futex_wake` = “Manager calls out: next 1 (or many) customers can come in!”

#### Sleeping Lock + futex

Futex Unlock

```C
void unlock(int val){
    if (atomic_dec(val) != 1){
        val = 0;
        futex_wake(&val, 1);
    }
}
```

Explanation:

- `atomic_dec(val)` → decrements lock state and returns old value.
- If old value was not 1, it means other threads are waiting.
- Set lock to free (`val = 0`) and wake one waiter.

Futex Lock

```C
void lock(int val){
    int c;
    if ((c = cmpxchg(val,0,1)) != 0){
        if (c != 2)
            c = xchg(val, 2);   // mark as "waiting threads"
        while (c != 0){
            futex_wait(&val, 2);
            c = xchg(val, 2);
        }
    }
}
```

Explanation:

- Try to atomically change 0 → 1.
  - If success → got the lock.
  - If fail →
    - Mark state as 2 = locked, waiting threads.
    - Sleep via futex_wait until woken up.
  - Once woken → retry.

#### Busy-Waiting vs Sleeping

- If lock is released very quickly → spinning is cheaper (avoid syscall overhead).
- If lock is held longer → better to sleep.

#### Two-Phase Waiting

- Worst case: spin too long when a context switch would’ve been cheaper.
- Solution: spin for at most C (≈ cost of context switch), then block.
  - Many systems use a two-phase lock:
    - Spin for a short while.
    - If still locked, fall back to futex sleep.

Analogy:

- Like waiting for an elevator:
- If you hear it ding soon → just wait.
- If it’s taking forever → sit on a bench and check your phone.

Guarantees performance within factor of 2 of optimal.

#### Synchronization Objectives

Mutual exclusion (A and B don’t run at the same time) → locks.

Ordering (B runs only after A) → semaphores or condition variables.

## Semaphore

A semaphore is an integer counter used for synchronization among threads/processes.

It has:

- Count (integer value)
- Queue of waiting threads (if they can’t proceed, they get suspended, not busy-wait).
- Three atomic operations only:
  - Initialize → sem_init
  - Wait / P (proberen = test) → sem_wait (decrement, block if < 0)
  - Signal / V (verhogen = increment) → sem_post (increment, wake up waiter if any)

Analogy:

Think of semaphores like parking spaces in a lot:

- count = number of free spots.
- When a car enters (sem_wait) → count goes down.
- If count < 0 → no spots → car must wait in a queue.
- When a car leaves (sem_post) → count goes up, and if cars are waiting, one gets in.

Sample Code:

```C
struct semaphore {
    int count;
    queue_type queue;
};

void sem_wait(semaphore s) {
    s.count--;
    if (s.count < 0) {
        // insert calling process to queue
        // block the calling process
    }
}

void sem_post(semaphore s) {
    s.count++;
    if (s.count <= 0) {
        // awaken one process from queue
        // move awakened process to ready queue
    }
}
```

Explanation:

- `sem_wait`
  - Decrement count.
  - If result < 0 → too many waiters → block caller.
- `sem_post`
  - Increment count.
  - If count ≤ 0 → there are blocked waiters → wake one.

> [!NOTE]
>
> - You cannot directly read/write semaphore value.
> - The value being negative means that many threads are waiting.

### Semaphore as Mutex

```C
sem_t mtx;
sem_init(&mtx, 1);   // start with 1 resource

sem_wait(&mtx);      // lock
// critical section
sem_post(&mtx);      // unlock
```

When initialized to 1, semaphore acts like a binary lock → mutex.

Analogy:

- One bathroom stall → `sem_init(1)`.
- Each person must `sem_wait` before entering.
- When leaving → `sem_post`.

### Semaphore for Ordering

```C
sem_t s;
sem_init(&s, 0);

thread_0() {
    S1;
    sem_post(&s);   // signal that S1 finished
}

thread_1() {
    sem_wait(&s);   // wait until S1 finished
    S2;
}
```

Analogy:
Think of a relay race: runner 2 waits for the baton (semaphore signal) before starting.

### Implementation Notes

`sem_wait` and `sem_post` themselves are critical sections, so they must be atomic.

Can be implemented using:

- Special machine instructions (atomic ops)
- Or with futex on Linux for efficiency.

### Types of Semaphores

Counting semaphore → initialized > 1 (e.g., multiple parking spots).

Binary semaphore (mutex) → initialized to 1 (e.g., one bathroom stall).

## Producer/Consumer Problem and the Bounded Buffer Problem

The Situation

- Producers: generate data and place it into a shared buffer.
- Consumers: take data out of the shared buffer.
- Buffer: has fixed size N → bounded.

The Challenges

- Consumer problem → consumer must not take from an empty buffer.
- Producer problem → producer must not put into a full buffer.
- Must ensure mutual exclusion while accessing the buffer.

Analogy:

- Think of a restaurant kitchen:
  - Chefs (producers) cook dishes and put them on a serving counter (buffer).
  - Waiters (consumers) pick dishes from the counter.
  - If the counter is full → chef must wait.
  - If the counter is empty → waiter must wait.

### Simple Case: One Producer, One Consumer, Buffer Size = 1

Use 2 semaphores:

- `emptyBuffer = 1` (initially one empty slot)
- `fullBuffer = 0` (no full slot initially)

Code:

Producer:

```C
while (1) {
    sem_wait(&emptyBuffer);  // wait until empty slot available
    fill(&buffer);           // put item
    sem_post(&fullBuffer);   // signal: buffer has item
}
```

Consumer:

```C
while (1) {
    sem_wait(&fullBuffer);   // wait until item available
    take(&buffer);           // consume item
    sem_post(&emptyBuffer);  // signal: buffer has empty slot
}
```

### Case 2: One Producer, One Consumer, Buffer Size = N

Now the buffer can hold N entries.

Semaphores:

- emptyBuffer = N (N empty slots)
- fullBuffer = 0 (no full slots initially)

Code:

Producer:

```C
i = 0;
while (1) {
    sem_wait(&emptyBuffer);      // wait for empty slot
    fill(&buffer[i]);            // add item
    i = (i + 1) % N;             // circular buffer index
    sem_post(&fullBuffer);       // signal item available
}
```

Consumer:

```C
j = 0;
while (1) {
    sem_wait(&fullBuffer);       // wait for filled slot
    take(&buffer[j]);            // consume item
    j = (j + 1) % N;             // circular buffer index
    sem_post(&emptyBuffer);      // signal empty slot available
}
```

> [!NOTE]
>
> - indices i (producer) and j (consumer) move circularly using modulo.
> - &emptyBuffer and &fullBuffer are counters not lists

### Case 3: Multiple Producers, Multiple Consumers

More complex: need mutex for critical section (to avoid race conditions on buffer
indices).

Semaphores:

- emptyBuffer = N
- fullBuffer = 0
- mutex = 1 (binary semaphore for mutual exclusion)

Code:

Producer:

```C
while (1) {
    sem_wait(&emptyBuffer);   // check for empty slot
    sem_wait(&mutex);         // lock critical section
    fill(&buffer[i]);
    i = (i + 1) % N;
    sem_post(&mutex);         // unlock
    sem_post(&fullBuffer);    // signal item ready
}
```

Consumer:

```C
while (1) {
    sem_wait(&fullBuffer);    // check for item
    sem_wait(&mutex);         // lock critical section
    take(&buffer[j]);
    j = (j + 1) % N;
    sem_post(&mutex);         // unlock
    sem_post(&emptyBuffer);   // signal slot free
}
```

### Things to Note

- emptyBuffer prevents overfilling.
- fullBuffer prevents underflow (taking from empty).
- mutex prevents race conditions when multiple producers/consumers update buffer
  indices.
- This is also called the Bounded Buffer Problem.

### Overall Analogy

- Buffer = serving counter.
- Producers (chefs) put dishes.
- Consumers (waiters) pick dishes.
- Semaphore emptyBuffer = how many empty plates left on counter.
- Semaphore fullBuffer = how many full plates are on counter.
- Mutex = only one person can touch the counter at a time (avoid chaos).

## Monitor

A monitor is a **programming language construct** that provides synchronization (like
semaphores).

Easier and safer to use because the compiler/runtime enforces the rules.

Implemented in languages such as:

- Concurrent Pascal, Modula-2, Modula-3, Java (in Java: synchronized keyword).
- A monitor = software module with:
  - Local data (shared resource)
  - Procedures (operations on the data)
  - Initialization code

Analogy:

Think of a bank vault. The vault (monitor) stores the data (money), and only official
bank procedures (authorized functions) can access it. No one can directly reach
inside.

### Characteristics of Monitors

- Encapsulation: local data accessible only via monitor procedures.
- Automatic mutual exclusion: only one process can execute inside the monitor at a time.
- Processes enter by calling a monitor procedure, and leave when the procedure ends.

Analogy:

Imagine a doctor’s office (the monitor). Patients (threads) can only interact with the
doctor by scheduling an appointment (procedure call). Only one patient can be inside at
a time.

### Problem: What if a process must wait?

A process inside the monitor may need to wait for some condition (e.g., “buffer not
empty”).

Since only one process can be active inside, we can’t let it just block there forever.

Solution: Condition Variables.

### Condition Variables

Special objects inside a monitor used for synchronization.

Support two operations:

- `cwait(c)` → the calling process is suspended until another signals condition c.
- `csignal(c)` → wake up one waiting process blocked on condition c.

Analogy:

- Like a waiting room outside the doctor’s office:
- If condition isn’t ready (doctor is busy), patient waits in the waiting room (cwait).
- When the doctor is free, receptionist calls one waiting patient (csignal).

### Structure of a Monitor

- Entry queue → processes waiting to enter.
- Condition queues (c1 … cn) → processes waiting for specific conditions.
- Urgent queue → processes waiting because another process signaled them.
- Shared data + procedures operate under automatic mutual exclusion.

Analogy:

- Think of it as a building with controlled doors:
- Entry queue = line outside the building.
- Condition queues = special waiting areas inside (e.g., “wait until X is done”).
- Urgent queue = priority line (a process got signaled and is about to re-enter).

### Monitors vs Semaphores

Semaphores: flexible but error-prone (easy to forget sem_post, or do sem_wait in wrong
order → deadlocks).

Monitors: safer, because the rules are built into the language/runtime.

## Pthread condition variables

There are just condition Variables let threads wait for certain conditions to become
true.

They are always used with a mutex lock (to avoid race conditions).

Unlike semaphores (which are counters), condition variables let you sleep until some
condition is met, without busy-waiting.

Core Operations:

1. pthread_cond_wait(&cond, &mutex)
   - Blocks the thread until cond is signaled.
   - While waiting, it releases the mutex (so other threads can change the condition).
   - When signaled, it reacquires the mutex before continuing.
2. pthread_cond_signal(&cond)
   - Wakes up one waiting thread on cond.
3. pthread_cond_broadcast(&cond)
   - Wakes up all waiting threads on cond.

> [!Important] Important rule:
> Always call these while holding the mutex lock.
> Otherwise, you risk lost wake-ups or inconsistent states.

### Why use while, not if?

In the Producer/Consumer code:

```C
while (numfull == N)  // producer waits if buffer is full
    pthread_cond_wait(&empty, &m);

while (numfull == 0)  // consumer waits if buffer is empty
    pthread_cond_wait(&full, &m);
```

Why `while`?

Because even after being woken, another thread may have changed the condition before
you got the lock again.

Using `while` re-checks the condition, avoiding bugs.

### Producer/Consumer with Condition Variables:

- Shared buffer with size N.
- numfull = how many slots currently filled.
- Two condition variables:
  - empty → signals producer when space is available.
  - full → signals consumer when an item is available.

Producer code (simplified):

```C
while(1) {
    pthread_mutex_lock(&m);

    while (numfull == N)      // buffer full
        pthread_cond_wait(&empty, &m);

    fill(&buffer[findempty()]);
    numfull++;

    pthread_cond_signal(&full);   // wake consumer
    pthread_mutex_unlock(&m);
}
```

Consumer code (simplified):

```C
while(1) {
    pthread_mutex_lock(&m);

    while (numfull == 0)      // buffer empty
        pthread_cond_wait(&full, &m);

    take(&buffer[findfull()]);
    numfull--;

    pthread_cond_signal(&empty);  // wake producer
    pthread_mutex_unlock(&m);
}
```

### Analogy

Think of it like a restaurant kitchen:

- Buffer = counter space with N plates.
- Producer (chef) = cooks meals and puts plates on the counter.
- Consumer (waiter) = takes plates from the counter to serve customers.
- Mutex = kitchen door — only one person can enter/exit at a time.
- Condition variables = bells:
  - “Counter has space!” → wakes up the chef.
  - “Food ready!” → wakes up the waiter.
  - Instead of pacing around (busy-waiting), the chef/waiter takes a nap until the bell rings.
