# Virtual Memory

## Support for Virtual Memory

Requirements:

- Hardware must support paging.
- OS must manage page movement between main and secondary memory.

## Valid–Invalid Bit

Each page table entry has a valid–invalid bit:

- v = in-memory.
- i = not in-memory.

If invalid during MMU translation → page fault

## Page Fault Handling

Steps:

1. Reference to missing page.
2. Trap to OS.
3. Page located on backing store.
4. Free frame found.
5. Bring missing page into frame.
6. Restart instruction.

## Average Memory Access Time (AMAT)

Formula:

AMAT = Tm + Pfault × Td

- Tm: memory access cost.
- Td: disk access cost.
- Pfault: probability of page fault.

Example:

- Tm = 100 ns, Td = 10 μs, Pfault = 0.1 → AMAT ≈ 1.1 μs.

Insight: Even small miss rates dominate due to slow disk

## Policies for Virtual Memory

Goal: Minimize page faults.

- Fetch Policy
  - Demand paging: Load page only when referenced → many initial page faults
  - Prepaging: Load additional pages proactively (efficient if contiguous, wasteful
    otherwise)

- Replacement Policy
  - Needed when all frames are full.
  - Principle: Replace page least likely to be used soon (locality).
  - Algorithms:
    - Optimal (OPT) – theoretical best, not practical.
    - Least Recently Used (LRU) – approximates OPT, but costly to implement.
    - FIFO – simple, but may evict frequently used pages.
    - Clock – uses reference bit, gives second chances.
    - Enhanced Clock – uses both reference and modify bits.

- Resident Set Management
  - OS decides how many pages per process.
  - Fixed: fixed number of frames per process.
  - Variable: frame allocation can change.
  - Scope:
    - Local – replacement within same process.
    - Global – replacement across all processes.

- Cleaning Policy
  - Demand cleaning: Write page only when replaced.
  - Precleaning: Write pages in batches

- Page Buffering
  - Maintain free frames and two lists (modified/unmodified).
  - Allows quick reuse and clustered writes.

## Linux Memory Management

Two aspects: Process memory space, Kernel memory space

Address Space:

- User mode cannot access kernel space.
- Kernel logical (contiguous) vs. kernel virtual (non-contiguous) addresses.

Address Translation:

- 4-level page table: 48-bit VA, 52-bit PA, 4KB pages.

Page Replacement:

- Active list = working set.
- Inactive list = replacement candidates.
- Uses Clock/LFU hybrid.

Page Cache:

- Caches file and block pages.
- Balances active/inactive lists.

Kernel Memory Allocator:

- Frequent small allocations.
- Uses buddy system + slab allocator.:w
