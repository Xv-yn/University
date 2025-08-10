# What is an Operating System?

An operating system is a software program. It’s not something physical you
can touch, but it’s essential software that acts like the manager or boss
of the computer. It controls both the hardware (like your CPU, memory,
hard drive, etc.) and software (like programs and apps).

The Operating System:

- Manages execution of programs
  - It decides which programs get to run and when.
- Controls and allocates memory
  - It decides how much RAM each program gets and keeps programs from
    interfering with each other’s memory.
- Manages input and output devices
  - This includes your keyboard, mouse, screen, printer, etc.
    The OS makes sure these devices talk properly with the programs.
- Manages files
  - It keeps track of where files are stored, how to read/write them,
    and enforces file permissions (e.g. who can open or edit a file).
- Facilitates networking
  - It helps your computer connect to other computers, like accessing
    the internet or shared files on a network.

Think of the structure of a whole computer as a top down sandwich, the more you learn
about programming, the more layers you can see and change:

- Applications (top bread) – What you use directly (games, browsers, etc).
- Libraries/Utilities – Make life easier for apps.
- Operating System (middle layer) – Manages everything.
- Hardware (bottom bread) – The physical machine.

```
    [ Applications ]
-------------------------
[ Libraries & Utilities ]
-------------------------
   [ Operating System ]
-------------------------
       [ Hardware ]
```

## Kernel vs Operating System

The Operating System (OS) and the Kernel are closely related, but they’re not
the same thing. Here's the difference:

- Kernel
  - The kernel is the core **part** of the operating system.
    - Think of it as the "brain" of the OS.
  - It directly interacts with the hardware.
  - It manages low-level tasks like:
    - CPU scheduling
    - Memory management
    - Device drivers
    - System calls (communication between programs and the OS)
  - The kernel is always running in memory (as long as the computer is on).
  - It acts as a bridge between applications and the hardware.

- Operating System
  - The OS includes the kernel plus all the other essential software that
    helps users and programs interact with the computer.
    - Think of the OS as the whole human body **including** the brain.
  - This includes:
    - The file system
    - User interface (GUI or command-line tools)
    - Network utilities
    - System libraries and services
  - The OS provides tools and environments to make the system usable.

If we think of it in terms of a car, the kernel would be the engine, and the
full OS is the car with steering, pedals, dashboard, and everything else you
need to drive it.

## Real World Example

On Linux:

- The kernel is literally called linux (e.g., vmlinuz file).
- The operating system might be Ubuntu, Fedora, or Debian—which includes the
  kernel, desktop environment, drivers, utilities, and package managers.

## Three Key Conceptual Responsibilities of an OS

1. Virtualization
   - Goal: Make each application think it has the computer to itself.
   - The OS creates the illusion that each program has full control of the CPU,
     memory, and other resources, even though they’re being shared with many other
     programs, by allocating virtual resources.
   - Example: When you open two browser tabs, each one feels like it’s
     running on its own, even though they’re actually sharing the same CPU
     and memory.
   - This makes things safer and more manageable.
2. Concurrency
   - Goal: Handle many things happening at once.
   - Multiple processes or threads can run "at the same time" (or appear to).
   - The OS must manage simultaneous events, like:
     - Your music player running while you type in a document.
   - Challenges:
     - Avoiding conflicts (e.g., two programs writing to the same file).
     - Ensuring fair use of CPU and memory.
3. Persistence
   - Goal: Keep data safe even after a crash or shutdown.
   - Data (like your saved files) must outlive the programs that created them.
   - Even if power is lost or the system crashes, your data shouldn't just
     vanish.
   - Challenges the OS faces:
     - Designing file systems with folders, files, and links.
     - Ensuring correctness: no data corruption if a crash happens during
       a save.
     - Overcoming performance limits (disks are slow!).

# Main Topics

The basic concepts (Fundamental Tasks) on which all operating systems are
built:

- Process management
- Memory management
- I/O device management
- File management
- Protection and security

## OS Control Tables

The OS needs to keep track of everything it manages — including processes,
memory, I/O devices, and files.

To do this, it uses tables:

1. Memory Tables
   - Track which parts of memory are used or free.
   - Record allocation to processes.
2. I/O Tables
   - Manage the status of I/O devices.
   - Track which processes are using which devices.
3. File Tables
   - Store info about open files, file permissions, and access status.
4. Process Tables
   - Contain a record for every active process (usually a Process Control
     Block for each).
   - Link to the process’s process image (the actual code, data, stack, etc.
     in memory).

# Process Management

## What is a Process?

A process is fundamental to how operating systems work. It can be understood at
different levels:

- A program in execution
  - A static program becomes a process when it's running.
- An instance of a running program
  - Multiple processes can come from the same program (e.g. opening Chrome
    twice = 2 processes).
- An entity assigned to a processor
  - The OS schedules this unit to run on the CPU.
- A unit with its own state, memory, and resources
  - It has a thread of execution, a current state (like running or waiting),
    and uses system resources (files, memory, etc).

In short:

- A process = a live, active, independently managed running program.

## Process Structure

Each process is uniquely defined by several attributes, which the OS uses to
manage it:

|    Attribute    |                                   Description                                   |
| :-------------: | :-----------------------------------------------------------------------------: |
|   Identifier    |                            Unique Process ID (PID).                             |
|      State      |                 Current status (Running, Waiting, Ready, etc.).                 |
|    Priority     |                      Helps decide the order of execution.                       |
| Program Counter |                 Keeps track of the next instruction to execute.                 |
| Memory Pointers |                 Where in memory the code and data are located.                  |
|  Context Data   | Register values, stack pointers, etc. (saved when switching between processes). |
| I/O Status Info |                      Info about files/devices being used.                       |
| Accounting Info |       CPU time used, memory used, etc., often for monitoring or billing.        |

> [!note]
> This is also called a Process Control Block (PCB)

A process attribute refers to any property or metadata the OS uses to
understand and manage the process. These are stored in the Process Control
Block (PCB).

## Process Attributes vs Process Image

The process image is the actual in-memory layout of everything the process
needs to run. It’s more about the contents than the metadata.

|          Component          |                       Description                        |
| :-------------------------: | :------------------------------------------------------: |
|        User Program         |               The actual executable code.                |
|          User Data          |          Modifiable variables and dynamic data.          |
|            Stack            | Holds function calls, return addresses, local variables. |
| Process Control Block (PCB) |      The metadata (aka attributes) of the process.       |

## Process Table

Referring back to the OS Control Tables...

The Process Table is the central hub — it tracks all active processes.

- Each process in the table can be linked to memory, I/O, and file resources.
- This means:
  - When a process is paused or resumed, the OS can reload its exact state.
  - If a process opens a file or uses memory, this is reflected and recorded
    in multiple tables.
- Tables are often cross-referenced, so data can be shared across subsystems.
- Example: A process's file descriptor might reference an entry in the file
  table.

## Creating Processes

When the OS decides to create a new process, it performs these steps:

1. Assigns a unique Process ID (PID)
   - This helps the OS track the process.
2. Creates and initializes a Process Control Block (PCB)
   - This stores metadata like PID, state, priority, program counter,
     registers, etc.
3. Allocates memory
   - Code, data, stack, and heap segments are set up in RAM.
4. Sets up linkages
   - For parent-child relationships, scheduling, or resource tracking.
5. Updates OS data structures
   - Includes process tables, scheduling queues, memory maps, etc.

Traditionally, the OS created all processes.

But in modern systems, it's common for one process to create another — this
is called process spawning.

Key Terms:

- Parent process: the process that creates another.
- Child process: the newly created process.

```
                       (init)
                       (pid=1)
                    /     |     \
                   /      |      \
                  /       |       \
                 /        |        \
                /         |         \
             (login)    (python)    (sshd)
            (pid=8415) (pid=2808) (pid=3028)
            /                           \
           /                             \
        (bash)                          (sshd)
      (pid=8416)                       (pid=3610)
       /      \                             \
      /        \                             \
   (ps)       (vim)                         (tcsh)
(pid=9298)  (pid=9204)                    (pid=4005)
```

The diagram shows a real-world tree of processes:

- At the top is init (pid = 1) — the very first process started by the OS.
- this init procress creates other processes like:
  - login, python, sshd
  - Those then create child processes, e.g.:
    - bash (from login)
    - ps and vim (from bash)
    - tcsh (from sshd)

## UNIX Process Creation

In UNIX, a new process is created by calling `fork()`.

This system call causes the OS (in kernel mode) to:

1. Allocate a slot in the process table.
2. Assign a PID to the new child process.
3. Copy the parent’s process image (code, stack, heap, etc.).
4. Update file counters to reflect shared open files.
5. Place the child into the Ready state.
6. Return values:
   - Parent gets child’s PID.
   - Child gets 0.

Once the child is created, the kernel’s dispatcher can do any of the following:

- Let the parent continue running.
- Let the child start executing immediately.
- Switch to another unrelated process entirely.

```
             parent                resumes
       .----------------> (wait) -------->
      /                     ^
fork()                      |
      \                     |
       '----> exec() ---> exit()
       child
```

The diagram shows this classic pattern:

- fork() → creates a child.
- In the child, call exec() → replaces the child’s memory with a new program.
- Meanwhile, the parent calls wait() → it pauses until the child finishes.
- After child’s exit(), the parent resumes.

In code:

```C
pid = fork();          // create child
if (pid < 0) {
    // Error
} else if (pid == 0) {
    // Child process
    execlp("/bin/ls", "ls", NULL);
} else {
    // Parent process
    wait(NULL);
    printf("Child Complete");
}
```

## Process Termination

How a Process Can Terminate

- HALT Instruction
  - The program executes a special instruction to stop.
  - This generates an interrupt to notify the OS.
  - Often used in system-level or bootloader code.
- User Action
  - The user manually quits the program or logs off.
  - Examples: Clicking "Close", Ctrl+C, or logging out of a session.
- Fault or Error
  - An unexpected condition like:
    - Division by zero
    - Segmentation fault
    - Illegal instruction
  - The OS usually kills the process to prevent system instability.
- Parent Process Terminates
  - The OS may automatically terminate the child processes.
  - Depends on the system — some UNIX systems allow orphans (which are
    adopted by init).

What Happens After Termination?

Once the OS knows a process has ended, it typically:

- Moves it to the "terminated" or "zombie" state briefly (to allow parent to
  read exit status).
- Then fully deletes the process from memory.

## Process States

The process state describes what a process is currently doing.

In the simplest model, a process can be either:

- Running (executing on the CPU)
- Not Running (waiting for its turn)

```
                          Dispatch
                 .-----------------------.
                /                         \
Enter          /                           v         Exit
------> (Not Running)                   (Running)------->
              ^                           /
               \                         /
                '-----------------------'
                           Pause
```

This diagram illustrates the life cycle of a process — specifically, how a
process transitions between different states as managed by the OS scheduler.
Here’s what each part means:

1. States
   - Not Running: The process is not currently using the CPU (it may be ready or waiting).
   - Running: The process is actively executing on the CPU.

2. Transitions
   - Enter: A process is created (e.g., by a user or system call) and enters
     the system.
   - Dispatch: The OS scheduler assigns the CPU to the process (switches it
     to "Running").
   - Pause: The process is interrupted or paused (e.g., by a time slice
     expiring, I/O request, or higher-priority process), and control is
     taken away — it returns to "Not Running."
   - Exit: The process finishes execution and leaves the system.

Main Purpose of the Diagram is to show how the OS activates and deactivates
processes (i.e., process state transitions).

More specifically:

- How the OS scheduler dispatches processes to the CPU.
- How a process can be preempted or paused and returned to the waiting state.
- The entry and exit points of a process lifecycle.

## Queuing Processes

```
                                       +-----------+
                                      /           /|   Exit
Entry        Queue        Dispatch   +-----------+-+------>
------> [[],[],[],[],[]] ----------> | Processor |/
    ^                                +-----------+
    |                                      |
    +--------------------------------------+
                      Pause
```

This diagram expands the idea of the "Not Running" state into a queue:

- Multiple processes line up in a queue waiting for CPU time.
- The dispatcher selects one process at a time and sends it to the processor.
- If it needs to wait or pause (e.g. waiting for I/O), it goes back to the
  queue.

This forms the basis of process scheduling.

## Accepted Model of Understanding

```
                       Dispatch
                    .-----------.
       Admit       /             v      Release
(New)--------->(Ready)        (Running)--------->(Exit)
                 ^   ^          /   |
                 |   '---------'    |
                 |    Timeout       |
            Event|                  |Event Wait
           Occurs|                  |
                 |                  |
                (Blocked)<----------+
```

This diagram adds more granularity with 5 states:

- New: Process is being created.
- Ready: Waiting in the queue for CPU time.
- Running: Currently executing on the CPU.
- Blocked: Waiting for an event (e.g. I/O completion).
- Exit: Process has completed or was terminated.

Realistically, processes don’t just run or not run — they go through
multiple stages like waiting for I/O, being created, or finishing. This
diagram reflects:

- How processes move from being created (New) to execution (Running).
- What happens when they’re waiting for something (e.g. blocked on I/O).
- What happens when the CPU stops executing them temporarily (Timeout).
- How they finally terminate (Exit).

It’s the first model that maps directly to how modern multitasking OSes
actually function.

It also explains how the OS manages resources.

Each state in the diagram represents a point where the OS is making important
scheduling decisions:

- Should this Ready process get CPU time next?
- Should the Running process be preempted (Timeout)?
- Is the Blocked process ready to return?
- Is the New process admitted to the system?

Label: Admit

- The OS accepts the new process into the system (e.g., from a program
  execution request).
- It’s now ready to be scheduled for the CPU.
- The process is placed in the ready queue.

Label: Dispatch

- The scheduler picks a process from the ready queue and assigns it the CPU.
- The process becomes active and starts executing instructions.

Label: Release

- The process finishes execution (normal completion or forced termination).
- All resources (memory, files, etc.) are released.
- It is removed from the system.

Label: Timeout

- If the process uses up its CPU time slice, the OS preempts it.
- It is returned to the ready queue to wait for another chance.
- Happens in preemptive scheduling.

Label: Event Wait

- The process requests an I/O operation or waits for an event.
- Since it can’t continue until the event finishes, it goes to the blocked (waiting) state.

Label: Event Occurs

- The event the process was waiting for has happened.
- Now the process is ready to run again and is placed in the ready queue.

(Not shown explicitly but implied)

- A process can stay in the ready queue indefinitely if it’s not chosen by the scheduler.
- This leads to scheduling problems like starvation.

Just like normal programming, the OS organises the processes into lists of
each category and each element in each list is a process.

- Running Processes
- Ready Processes
- Blocked Processes

## Sample OS Step-by-Step Process Management

1. User process is running.
2. Timer interrupt or system call occurs → Trap/Interrupt → OS regains control.
3. OS consults the scheduler → decides who should run.
4. OS uses the dispatcher to switch context and start the chosen process.

## Process Dispatching

Once the OS has control, how does it decide which process to run?
It does this by using a scheduler and dispatcher:

- Scheduler: Decides which process should run next, based on:
  - Priorities
  - Time used
  - I/O patterns
- Dispatcher: Implements that decision by:
  - Saving the current context
  - Loading the new process's context
  - Jumping to the new process

## Process Interrupts

How does the OS guarantee it gets control back periodically?

- Hardware timer interrupt is configured to fire at regular intervals (e.g.,
  every 10ms).
- When the interrupt fires, the OS preempts the current process, regains
  control, and can make scheduling decisions.

How does the OS regain control from a running user process?"

- Traps (Internal Events): These are triggered by the process itself, such as:
  - System calls (e.g., file read/write)
  - Errors (e.g., illegal memory access)
  - Page faults (accessing data not in RAM)
- Interrupts (External Events): Triggered outside the user process, such as:
  - Keyboard input
  - Disk completion
  - Timer interrupts (crucial for preemption)

## Switching Process States

When a process stops running, the OS saves everything the CPU was using for
that process into its Process Control Block (PCB):

- Program Counter (PC) – address of next instruction
- Processor Status Word (PSW) – flags and condition codes
- General-purpose registers – used for variables, memory references, etc.

And to switch the current process, this is what the OS does:

1. Save the current process’s context (its CPU state).
2. Update its PCB with that context.
3. Move it to a queue (like Ready or Blocked).
4. Choose another process to run (using the scheduler).
5. Update that process’s PCB to reflect it's now running.
6. Update memory management info, if needed.
7. Restore the new process’s saved context into the CPU.

Most CPUs have built-in instructions for quickly saving/restoring minimal
state (e.g., PC and PS).

- The OS is responsible for saving the full process context in the PCB.
- Full context switching is expensive, so:
  - Sometimes only partial state is saved (like during quick interrupts).
  - Some OSes do incremental saving to reduce overhead.

Think of it like a save/load system in a video game — only instead of saving
your sword and location, it’s saving CPU registers and instruction addresses.
