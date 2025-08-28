# Concurrency: Deadlock

## Deadlock

A deadlock is when a set of threads/processes are all blocked forever because each one
is waiting for an event (usually a lock/resource) that another blocked thread must
provide. No one can move, so the program stalls.

Example:

Timeline

1. T1 acquires A.
2. T2 acquires B.
3. T1 now waits for B (held by T2).
4. T2 now waits for A (held by T1).

Both wait forever ⇒ deadlock.

## Resource allocation graphs

A deadlock visualization in the form of graphs

```
    +-------------------------------+
    |                               |
    v                               |
(Thread 1)  (Thread 2)  (Thread 3)  |
    |      .-^  |      .-^  |       |
    |     /     |     /     |       |
    |    /      |    /      |       |
    v   /       v   /       v       |
 [Lock A]    [Lock B]    [Lock C]---+
```

- Thread 1 Locks A
- Lock A depends on Thread 2
- Thread 2 Locks B
- Lock B depends on Thread 3
- Thread 3 Locks C
- Lock C depends on Thread 1

## Necessary conditions for deadlock

All four must hold:

- Mutual exclusion – resource can’t be shared simultaneously.
- Hold and wait – a thread holds one resource while waiting for another.
- No preemption – you can’t forcibly take a resource away.
- Circular wait – there’s a cycle of threads each waiting on the next.

Break any one, and deadlock can’t occur.

### Dining philosophers problem

N philosophers sit around a round table.

N chopsticks (one between each pair of philosophers).

Each philosopher alternates: think → get both chopsticks → eat → put chopsticks down.

Deadlock: If everyone picks up their left chopstick first, then each waits for the
right chopstick (held by their neighbor). No one can proceed → all block forever.

## Preventing a Deadlock

- Impose a global lock order: always acquire locks in the same order (e.g., sort by
  address/id).
- All-or-none (try-lock + backoff): try to grab all needed locks; if any try fails,
  release what you hold, wait, and retry.
- Use combined/Scoped locking primitives: e.g., acquire multiple locks atomically
  (std::scoped_lock in C++, ReentrantLock.tryLock patterns in Java).
- Time out & recover: abandon after a timeout, roll back, and retry.
- Reduce granularity / single lock: when acceptable, protect the critical region with
  one lock.
- Keep critical sections small: don’t call out or block while holding locks.
