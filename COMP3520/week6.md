# Process Schedulers

## Linux O(1) Scheduler

- Used before Linux 2.6.23.
- How it works:
  - Two arrays: active and expired, each with multiple priority queues.
  - The CPU always picks the highest priority runnable task from the active array.
  - Once a task uses up its time slice, it moves to the expired array.
  - When the active array is empty, the two arrays are swapped.
- Purpose: Prevent starvation (low-priority tasks never running).

### Structure of O(1) Scheduler

- Priorities range from 0 (highest) to 139 (lowest).

- Real-time tasks get longer time slices (e.g., 200ms) while lower-priority tasks may
  only get 10ms.
  - Real-time tasks are processes that must meet strict timing requirements.
    - Example: Playing audio, controlling a robot arm, or handling signals in a
      pacemaker.
    - If they don’t run on time, things break (e.g., sound glitches, robot misses a
      step).

  - Real-time tasks (priority 0–99): get longer slices (like 200ms).
    - Reason: They need to finish quickly and reliably.
  - Normal/background tasks (priority 100–139): get shorter slices (like 10ms).
    - Reason: They aren’t urgent, so it’s fine to preempt them more often to keep
      the system responsive.

- Efficient search:
  - Each priority has a bit in a bitmap.

  ```
  Priority:   0  1  2  3  4  5 ...
  Bitmap:     0  1  0  0  1  0 ...
  ```

  - This means:
    - Priority 1 has tasks waiting.
    - Priority 4 has tasks waiting.
    - Others are empty.

  - Analogy: Think of it like a row of light switches. Each switch = a priority
    level. If a switch is ON (1), there’s work waiting at that level. The CPU
    doesn’t need to peek into every room—it just looks for the first ON switch.

  - CPU finds the highest-priority ready task by checking the first set bit (fast
    and hardware supported).

- Time slice calculation:
  - For high priority (static priority < 120): longer quanta.
  - For low priority (≥120): shorter quanta.
  - Example:
    - Nice = -20 → 800ms
    - Nice = 0 → 100ms
    - Nice = +20 → 5ms

> [!NOTE]
> Niceness is another way of saying priority
> :ower Nice = Higher Priority

- Dynamic Priority:
  - Based on static priority + sleep bonus.
  - Interactive tasks (which sleep often) get boosted.
  - Helps responsiveness for user-interactive programs.

> [!Warning] Issue
> The O(1) scheduler became complex (hard to tune starvation limit, interactive
> identification), so it was replaced.

## Linux CFS (Completely Fair Scheduler)

- Introduced: Since Linux 2.6.23.
- Goal: Fair CPU sharing among tasks, especially improving interactive performance.
- Core Idea:
  - Divide CPU fairly among all tasks, weighted by priority (niceness).
  - Introduces virtual runtime (vruntime):
    - Tracks how much CPU a task has used (normalized by weight).
    - The task with the lowest vruntime gets scheduled next.

### How CFS Works

- Default scheduling window: 48ms (sched_latency).
- If some finish, remaining tasks get larger slices (fair division).

- E.g. If 4 tasks exist, each gets 12ms.

> [!Note]
> Min granularity: A task’s slice won’t go below ~6ms to avoid excessive context
> switches.

Priorities in CFS:

- 40 levels, nice values from -20 (highest) to +19 (lowest).
- Each maps to a weight (Like a dictionary where each key leads to an array of tasks).
- Higher priority (lower nice) → larger weight → more CPU time.

- Time slice formula:

  ```
  time_slice(task) = (task_weight / total_weight) * sched_latency
  ```

- Vruntime update:
  ```
  vruntime += (weight0 / task_weight) * actual_runtime
  ```

> [!NOTE]
> (weight0 = baseline weight for nice=0)

Implementation detail:

- CFS uses a red-black tree.
- The leftmost node = task with lowest vruntime → chosen next.
- Operations: O(log N) insertion/deletion, O(1) selection.

When the task runs:

- Its vruntime increases (because it just consumed CPU time).
- The scheduler removes it from the tree, updates its vruntime, then reinserts it into
  the tree.
- Because its vruntime is now bigger, it usually shifts to the right side of the tree

## Summary on Scheduling

Types of scheduling decisions:

- Long-term: Which jobs to admit into system.
- Medium-term: Which jobs to suspend/resume.
- Short-term: Which process to run next.

Goals:

- From user’s view: response time, turnaround time.
- From system’s view: throughput, CPU utilization.

Algorithms covered: FCFS, SJF, SRT, RR, MLFQ.

Key lesson: No scheduler is universally best; choice depends on workload and hardware.

## Priority Inversion

- Definition: A high-priority task waits because a low-priority task holds a resource
  it needs.

- Problem: Even medium-priority tasks can delay the low-priority one, making the
  high-priority task wait indefinitely.

- Famous case: NASA’s Mars Pathfinder.

- Example situation:
  - T1 (high) is blocked because T3 (low) holds a lock.
  - T2 (medium) preempts T3, causing unbounded waiting for T1.
  - Priority Inheritance
    - Fix: The low-priority task temporarily inherits the high priority of the
      blocked task.
    - This ensures it runs quickly and releases the resource.
