# File System Consistency

Crash-Consistency Problem

When a system crashes during an update, on-disk structures can become inconsistent.
A consistent file system ensures data structures always remain valid even after
crashes.

Approache to this problem:

- File System Checker (fsck):
  - Scans the disk to find contradictions and repairs inconsistencies.
  - Very slow and expensive since it checks the entire disk.
  - Only restores consistency, not necessarily the correct data.
- Journaling:
  - Records intended changes in a journal (log) before applying them.
  - Allows faster recovery by replaying completed transactions and skipping
    incomplete ones.

## Journaling (Write-Ahead Logging)

Concept

- Write pending data/metadata to a journal (log).
- After writing completes → commit.
- Then, apply updates to their real locations (checkpoint).
- After a crash, recover using the committed transactions from the journal.

Transaction Process

- TxB (Transaction Begin) and TxE (Transaction End) mark transaction boundaries.
- Data and metadata updates are written atomically to prevent half-written data.
- The three-phase protocol ensures atomic updates:
  - Journal Write
  - Journal Commit
  - Checkpoint

Metadata Journaling

- Logs only metadata (not data blocks) to save I/O cost.
- Common in ext3/ext4 for efficiency.

Optimizations

- Batching Log Updates: Combine multiple file changes into one global transaction.
- Circular Log: Reuse journal space after transactions are checkpointed.

## Linux File Systems

ext2

- Based on the UNIX Fast File System.
- Uses fsck (no journaling).

ext3

- Adds journaling with three modes:
  - Data Mode: Logs both data & metadata (safest, but slowest).
  - Ordered Mode: Logs metadata after writing data (balanced).
  - Writeback Mode: Logs only metadata (fastest, riskiest).
- Adds HTree directory indexing for large directories — faster lookups.

ext4

- Replaces indirect addressing with extents (continuous block regions).
- Supports huge files and volumes (up to exabytes).
- Features for performance and fragmentation control:
  - Multiple block allocation
  - Pre-allocation
  - Delayed allocation

## ZFS (Zettabyte File System)

Motivation

ext file systems have drawbacks:

- Manual capacity management
- Limits on volume/file size
- Slow fsck and journaling
- No defense against silent corruption

Key Features

- Pooled Storage Model:
  - No fixed partitions; all disks share space dynamically.
- Copy-on-Write (CoW):
  - Never overwrites data in place.
  - Writes go to free blocks, then pointers are atomically updated.
  - Ensures transactions are either fully committed or ignored → always consistent.
  - No need for journaling.

Snapshots:

- Cheap and fast to create.
  - Enables quick recovery and backups.
  - Keeps old data versions intact until explicitly deleted.
- Checksumming:
  - ZFS stores checksums in parent blocks, forming a Merkle Tree.
  - Detects and self-heals corrupted data via redundancy.
