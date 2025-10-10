# I/O System Organization

This is how the operating system (OS) connects applications to hardware devices
through layers.

Layers Explained:

- Application Process:
  - The program or user process that requests I/O (e.g., reading a file,
    printing a document).
  - It uses system calls like read(), write(), or open().
- File Manager:
  - Manages files and data organization.
  - Handles user-level file operations and passes them to the appropriate driver.
- Device Driver:
  - The middle layer that converts general I/O commands into device-specific
    instructions.
  - Split into two parts:
    - Device-Independent: Handles common I/O functions (buffering, error
      handling, naming).
    - Device-Dependent: Specific to each device (e.g., printer vs. keyboard).
- Device Controller (Hardware):
  - The actual physical component that sends commands to the device and returns
    status/data to the OS.
  - Contains registers for Command, Status, and Data I/O.

Analogy:

Think of it as a chain of communication:

App → File Manager → Device Driver → Device Controller → Hardware Device.

Each layer translates the request into something the next layer can understand.

# Polling I/O Read Operation

Polling is a basic method of I/O where the CPU keeps checking (“polling”) the device
to see if it’s ready to send or receive data.

Steps:

1. The application calls read(device, …) to get data.
2. The read request goes to the read function in the device driver.
3. The driver sends a command to the device controller to start the I/O operation.
4. The CPU polls (repeatedly checks) the Status Register in the controller to see if the operation is done.
5. When the device is ready, the data is transferred back to the application.

Key Point:

Polling wastes CPU time because it must keep checking the device instead of doing
other tasks.

# Interrupt-driven I/O Read Operation

This is a smarter method — the CPU doesn’t keep checking the device. Instead, the
device notifies the CPU when it’s ready using an interrupt signal.

Steps:

1. The application calls read(device, …) (same as before).
2. The OS sets up the read function and device command.
3. The device controller starts reading.
4. When it’s done, it sends an interrupt signal to the CPU.
5. The Interrupt Handler in the OS takes control.
   - It identifies the device that caused the interrupt.
   - The Device Handler processes the completed I/O and updates the Device Status
     Table.
6. The data is then returned to the application.

Key Advantage:

The CPU is free to do other work while waiting for I/O. It only responds when
interrupted, making this method much more efficient than polling.

# DMA (Direct Memory Access) Transfer

DMA is the most efficient I/O method — it allows devices to transfer data directly to
or from memory without involving the CPU for every byte.

Steps:

1. The device driver tells the DMA controller where in memory to place the data.
2. The driver instructs the disk controller to begin the transfer.
3. The disk controller sends data to the DMA controller.
4. The DMA controller writes the data directly into memory (buffer X).
5. The DMA controller automatically updates memory addresses as data is transferred.
6. When done, the DMA controller interrupts the CPU to say “transfer complete.”

Key Benefits:

- Frees up the CPU for other tasks.
- Great for large, continuous data transfers (e.g., copying files, streaming video).

# Magnetic Disks (HDD)

Magnetic disks are traditional hard drives used for secondary storage (long-term data).

How it works:

1. Disks consist of platters (circular disks) that spin around a spindle.
2. Each platter has tracks (concentric circles) divided into sectors (small pieces).
3. Multiple platters share the same arm assembly that moves read/write heads to the correct track.
4. Data is read as the disk rotates under the head.

Performance factors:

- Seek time – how long the head takes to move to the right track.
- Rotational delay – time waiting for the right sector to spin under the head.
- Transfer time – actual time to read/write data.

Concept:

Access time = Seek time + Rotational delay + Transfer time.

# Flash-based SSD (Solid-State Drive)

SSDs store data in flash memory chips — no moving parts, much faster than HDDs.

Components:

- Flash chips – store the actual data (like cells on a grid).
- Volatile memory (SRAM) – temporary memory for caching data and storing mapping
  tables.
- Flash Controller – manages data movement and coordinates between the CPU and flash
  memory.
- Interface Logic – connects the SSD to the computer system.

Key Traits:

- Faster read/write speeds.
- Lower power use.
- No noise or mechanical failure risk.

But: limited write/erase cycles (wear-out problem).

# Flash Translation Layer (FTL)

The FTL is the “brain” of an SSD. It converts logical block addresses (used by the OS)
into physical locations on flash memory.

How it works:

1. The application and file system send read/write requests like “save this file at
   block 5.”
2. The FTL translates this into flash-level commands (read, erase, program).
3. Underneath, the Flash Memory Controller handles the actual electrical writing.

Why it’s needed:

- You can’t overwrite data in flash memory — you must erase a whole block first, then
  write again.
- The FTL makes this process invisible to the user.

# FTL Internals: Wear Levelling & Mapping Table

To prevent SSD damage and maintain speed, the FTL uses wear levelling and mapping
tables.

Wear Levelling:

- Every erase/write wears out flash cells.
- The FTL spreads writes evenly so no block dies early.

Log-structured Writing:

- SSDs never overwrite old data — instead, they append new data to a free block.
- The old data becomes garbage, which will be cleaned up later.

Mapping Table:

- Keeps track of which logical block (the one OS sees) maps to which physical block
  (on the SSD).
- Updated every time new data is written.

Example:

1. New data replaces old data at another location.
2. Old block marked as “garbage”.
3. Mapping table updated to point to the new block.

# Mapping Table Example

Shows how the SSD records where each piece of data is stored.

Steps:

1. Initially, all pages are marked invalid (i) — nothing is stored yet.
2. The SSD issues an erase command to prepare the block (changes state to E).
3. New writes occur:
   - Write(100) → a1
   - Write(101) → a2
   - Write(2000) → b1
   - Write(2001) → b2
4. The mapping table stores which logical blocks (100, 101, etc.) correspond to
   which physical pages (0, 1, 2…).

Analogy:

It’s like a table of contents in a book — the OS says “give me page 100,” and the SSD
looks up where that actually is in memory.

# RAID

RAID is a method of combining multiple physical hard drives into one logical storage
unit to improve speed, reliability, or both.

Think of it like teamwork among disks — instead of one person (disk) doing all the
work, a group of people (disks) work together so:

- If one person drops a paper (a disk fails), others can still recover it.
- If everyone writes different parts at the same time, the job finishes faster.

RAID stands for Redundant Array of Independent Disks.

> [!NOTE]
> RAID is a storage organization method, meaning that it can be used with any
> kind of storage drive — including HDDs, SSDs,

It’s a data storage virtualization technology that merges multiple physical drives
into a single logical unit. Depending on configuration, RAID can:

- Increase performance (via parallel data access)
- Provide fault tolerance (via redundancy or parity)
- Or balance both.

When you save data, RAID splits or duplicates it across several disks. How it does
this depends on the RAID level (0, 1, 5, etc.).

It may:

- Stripe data (split across disks for speed)
- Mirror data (copy to multiple disks for safety)
- Add parity information (a checksum-like backup that helps rebuild data if a disk
  fails)

| **RAID Level**         | **What It Does**                            | **Pros**                                      | **Cons**                                            | **Min. Disks** |
| ---------------------- | ------------------------------------------- | --------------------------------------------- | --------------------------------------------------- | -------------- |
| **RAID 0 (Striping)**  | Splits data evenly across disks (no backup) | Fast performance                              | No protection — if one disk fails, all data is lost | 2              |
| **RAID 1 (Mirroring)** | Copies data to both disks                   | High reliability (redundancy)                 | Uses 2× storage space                               | 2              |
| **RAID 4**             | Striping + 1 dedicated parity disk          | Good read speed, fault tolerance              | Parity disk can become a bottleneck                 | 3              |
| **RAID 5**             | Striping + distributed parity across disks  | Great balance between speed & fault tolerance | Slower writes (due to parity calc)                  | 3              |
| **RAID 6**             | Like RAID 5 but with 2 parity blocks        | Can survive 2 disk failures                   | Even slower writes                                  | 4+             |
| **RAID 10 (1+0)**      | Mirroring + striping                        | High speed + reliability                      | Expensive (half storage used)                       | 4              |

Key concepts:

- Striping: Splitting data across disks for parallel reads/writes → faster performance.
- Mirroring: Duplicating data for fault tolerance → if one disk dies, data survives.
- Parity: Mathematical backup used to reconstruct lost data.

Trade offs:
| Goal | Best RAID Option | Trade-off |
| ------------------------ | ----------------- | ----------------- |
| Speed only | RAID 0 | No safety |
| Safety only | RAID 1 | Uses more space |
| Balance (speed + safety) | RAID 5 or RAID 10 | More disks needed |
| Max reliability | RAID 6 | Slower writes |
