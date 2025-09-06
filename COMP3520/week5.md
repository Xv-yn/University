# Processor Scheduling

## Types of Scheduling

1. Long-term scheduling
   - Decides which programs are admitted for processing.
   - Controls degree of multiprogramming (too many processes = less CPU share).
   - Runs infrequently (minutes, maybe longer).
   - Decides which jobs enter the “office” (memory + ready queue).
   - Example: Out of 50 jobs waiting on disk, admit 10 so the system isn’t overloaded.

2. Medium-term scheduling
   - Part of swapping function.
   - Manages degree of multiprogramming by swapping in/out processes, considering
     memory needs.
   - Runs occasionally (seconds–minutes).
   - Suspends/resumes jobs if memory or CPU is overloaded.
   - Example: If a heavy process is hogging resources, swap it out for a while.

3. Short-term scheduling (Dispatcher)
   - Most frequent.
   - Invoked on events: clock interrupts, I/O interrupts, OS calls, signals.
   - Runs very frequently (milliseconds).
   - Decides which process in the ready queue gets the CPU next.
   - Uses algorithms like Round Robin, FCFS, SJF, Priority, MFBQ.

## Scheduling Criteria (What makes a good scheduler)

- Turnaround Time = Completion time – Arrival time
  - Includes execution + waiting time.
  - Example: Process arrives at 10, finishes at 30, needs 15 execution → Turnaround
    = 20, Wait = 5.

- Response Time = First run time – Arrival time
  - Important for interactive systems.

- Other objectives: fairness, deadlines, and efficiency.

## Scheduling Policies/Algorithms

1. Non-Preemptive
   - FCFS (First-Come-First-Served)
     - FIFO (First In First Out) order.
     - Simple, but short jobs may wait long.
     - Favors CPU-bound processes, I/O-bound suffer.

   - SJF (Shortest Job First / Shortest Process Next)
     - Picks shortest job.
     - Better average turnaround & waiting times.
     - Problems: starvation of long jobs, needs job length prediction.

2. Preemptive
   - SRT (Shortest Remaining Time)
     - Preemptive SJF (Shortest Job First).
     - Always run job with shortest remaining time.
     - Improves turnaround but risks starvation.
   - RR (Round Robin)
     - Uses time slices (quantum).
     - Improves response time, but may increase turnaround with many long jobs.
     - Trade-off: time slice too small → high context-switch overhead; too large →
       like FCFS.

3. MFBQ (Multilevel Feedback Queue)
   - Multiple queues with priorities.
   - Rules:
     - Highest priority jobs run first.
     - Equal priority jobs → Round Robin.
     - Jobs start at high priority, demoted if they use full time slice.
   - Benefits: balances short vs long jobs without knowing lengths.
   - Issues: starvation of low-priority jobs.
   - Solution: periodically boost all jobs to top queue.

## Contemporary Scheduling

- Based on time quantum + preemption.
- Priority-driven job selection.
- Often implemented as a variant of MFBQ for real systems.
