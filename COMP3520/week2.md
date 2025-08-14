# Dual Mode Operations

We split the "editing" modes into two different modes. We do this to protect the OS
kernel and critical resources from accidental or malicious interference by user
programs.

The two modes are as follows:

- User Mode:
  - Regular applications run here.
  - Limited access to system resources (e.g., can’t directly access hardware or
    change memory mapping).
  - Certain CPU instructions are privileged and cannot be executed.

- Kernel Mode (Supervisor Mode):
  - Operating system core runs here.
  - Full access to hardware and system resources.
  - Can execute privileged instructions.

> [!NOTE]
> Kernel mode is NOT like `sudo`. `sudo` is mainly used to download things and perform
> privileged user acctions, wheras kernel mode is a CPU execution state with
> unrestricted hardware and memory access.
>
> Think of it as having the keys to a building (sudo) and having the blueprints and
> the ability to rebuild the building’s wiring and plumbing (kernel mode).

## System Calls

Instead of giving user programs direct access to kernel mode, the operating system
provides system calls — a safe, controlled interface to request kernel services.

Think of it like an API, but for the operating system.

When a program needs something privileged (e.g., read a file, create a process, access
hardware), it makes a system call. The CPU then switches from user mode to kernel mode,
executes the requested service, and then switches back to user mode.

This way, the user program never runs in kernel mode directly — it just asks the
kernel to do things on its behalf.

A more detailed step by step is as follows:

1. Save user registers
   - Kernel bookmarks your program’s spot.
   - Analogy: Receptionist asks you to stand exactly where you are and cannot move.

2. Read syscall number
   - Figures out which service you want (read, write, open...).
   - Analogy: Receptionist reads the service code on your ticket.

3. Check arguments
   - Validates pointers, sizes, permissions.
   - Analogy: Receptionist confirms if your ticket is correct.

4. Do the operation
   - Kernel performs the privileged work.
   - Analogy: Receptionist gives the ticket to a technician, who goes into the
     staff-only back room and does it.

5. Store result / set return value
   - Places status/bytes-written/FD, etc., where your program will read it.
   - Analogy: The technician puts the finished package + receipt on the counter.

6. Return-from-trap (switch back)
   - Restores user mode, stack pointer, program counter, registers.
   - Either returns to you or lets the scheduler run someone else.
   - Analogy: The receptionist gives you the package and you are now free to move again.

# Processes and Threads

A process is like a self-contained mini operating system for its own program:

- It has its own memory space, resources, and permissions.
- It keeps track of its own files, variables, and execution state.

A thread inside that process is like an individual worker in this mini-OS:

- Threads share the same memory and resources of the process (like workers sharing the
  same office and tools).
- Each thread can run tasks independently, but changes made by one worker (thread) are
  visible to all other workers in that office (process).

Similar to processes, threads also have execution states that describe what they are currently doing:

- Running – Actively executing on the CPU.
- Ready – Waiting in the ready queue for CPU time.
- Blocked – Waiting for some event (e.g., I/O completion, resource availability).

Threads can move between these states through certain operations:

- Spawn – Create a new thread, allocating its register context and stack.
- Block – Move to a waiting state until an event occurs.
- Unblock – Move back to the ready queue once the event is complete.
- Finish – End execution and free resources.

Analogy:
Think of each thread as a worker in an office:

- Running – The worker is actively doing a task at their desk.
- Ready – The worker is standing by, waiting for their turn to use a shared resource.
- Blocked – The worker is waiting for someone to hand them the documents they need.
- Finish – The worker clocks out and leaves the office.

In Linux, processes and threads are represented internally in the same way — as tasks.
The difference lies in how much of the execution context they share:

- `fork()` – Creates a new process with its own completely separate memory space, file
  descriptors, and resources. It’s an independent copy of the parent.
- `clone()` – Creates a new task (which could be a thread) with its own task identity,
  but allows sharing specific resources (e.g., address space, file descriptors,
  signal handlers) with the parent.

When `clone()` is used with the right flags to share the address space, the result is
a thread rather than a fully independent process.

Analogy:

- `fork()` → “Make a photocopy of the whole office, including desks, files, and staff.”
- `clone()` → “Hire another worker to work in the same office, sharing the same files
  and desk, but still having their own ID badge.”

# Concurrency

## Race Condition

A racecondition is when two or more threads/processes try to access and modify shared
data at the same time, and the final result depends on the timing of execution.

Analogy: Two people editing the same document at the same time without coordination —
one’s changes can overwrite the other’s.

## Critical section

A critical section is a block of code where shared resources are accessed.

Only one thread/process should be allowed here at a time to avoid race conditions.

Analogy: Only one person can be inside the “vault” at a time to keep the money safe.

## Mutual exclusion

A mechanism to ensure that only one thread/process enters the critical section at a
time.

Achieved with locks, semaphores, or other synchronization tools.

## Hardware support

Some CPUs provide built-in instructions to make mutual exclusion easier and faster.

### Atomic operations

Operations that happen completely or not at all — no interruptions.

Analogy: Snapping your fingers — it’s instant from the system’s point of view.

### Special machine instructions

**Compare & Swap (CAS)**: Checks if a memory location has an expected value; if yes,
changes it.

**Exchange**: Swaps the contents of a memory location with a register atomically.

These let us implement locks without needing heavy OS intervention.
