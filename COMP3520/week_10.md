# Introduction to File Management

File: A collection of related data that can be read and written.

File System: Provides an organized way to store, access, and manage files.

Components:

- Logical File System: User-facing; manages APIs for access, directories, and
  protection.
- Physical File System: Handles low-level operations like block allocation and disk I/O.

Goals:

- Provide a standardized I/O interface
- Manage secondary storage
- Optimize performance and reliability
- Prevent data loss or corruption

## File Structure and Attributes

Logical File Structures:

- Flat file of bytes
- Fixed-length or variable-length records
- Most modern OSs: treat files as linear byte arrays

File Attributes:

- Name: Human-readable identifier
- Identifier: Unique system tag (e.g., inode number)
- Type, Location, Size, Protection
- Timestamps & User ID: For security and tracking
- Stored in a File Control Block (FCB) or inode (in UNIX/Linux)

## File System Organization

A simple file system uses:

- Superblock: Metadata about the file system (layout, counts, etc.)
- Bitmaps: Track free/used data blocks and inodes
- Inode Table: Stores metadata for files (e.g., pointers to data blocks)
- Data Blocks: Store actual user data

Example layout (64 blocks, 4KB each):

```txt
[ Superblock | Bitmaps | Inodes | Data Blocks ]
```

Each inode = 256B → 16 inodes per 4KB block → 80 inodes (≈ 80 files)

## UNIX Inode System

Each inode contains:

- File mode, ownership, timestamps
- Direct, single indirect, double, and triple indirect pointers

Capacity (4KB block size):

| Level           | # Blocks | Capacity |
| --------------- | -------- | -------- |
| Direct          | 12       | 48 KB    |
| Single Indirect | 512      | 2 MB     |
| Double Indirect | 256K     | 1 GB     |
| Triple Indirect | 128M     | 512 GB   |

## Directories

Directories are special files containing filename–inode pairs.

- Each directory has its own inode.
- Can only be modified indirectly (e.g., by creating/deleting files).
- Directory structure forms an acyclic tree; allows duplicate filenames in different
  directories.

## File System API (UNIX/Linux)

Common calls:

```C
open(), read(), write(), close(), unlink(), lseek()
opendir(), readdir(), closedir(), rmdir()
```

## Access Methods

Every file/directory has an inode number (inumber).

To access a file, find its inode using the inumber → locate data blocks.

Example: Blocks 3–7 hold inodes; use inumber to find correct block and position.

## File Operations

- `open()`
  - Returns a file descriptor (fd) — a reference to an inode in memory.
  - Stored in a per-process fd table for fast access.
  - Avoids repeated traversal of long pathnames.

- Example traversal for /foo/bar:
  1. Read root inode
  2. Read root directory (find `foo`)
  3. Read `foo` inode & directory (find `bar`)
  4. Read `bar` inode

- `read()` & `write()`
  - After opening, inode remains in memory → fewer disk I/Os.
  - Read: load data block.
  - Write: update block, inode, and possibly bitmaps.

- `create()`
  - More expensive — needs new inode, data block, and directory updates.

## Caching and Buffering

Frequent disk access → slow.

Use main memory (DRAM) to cache important blocks.

- Early FS: fixed 10% cache; used LRU replacement.
- Modern FS: unified page cache integrates virtual memory + file cache dynamically.

## File Layout and Performance

Simple FS (not disk-aware):

- Treats disk as random-access memory.
- Data blocks may be far from inodes → fragmentation.

Fast File System (FFS):

- Disk-aware layout for performance.
- Organizes disk into cylinder groups — clusters of nearby tracks.

## Placement Techniques

Cylinder/Block Groups:

- Each group contains:
  - Duplicate superblock
  - Bitmaps
  - Inode table
  - Data blocks
- Goal: keep inodes close to data for faster access.

Placement Rules:

- Directories: place in groups with few directories + many free inodes.
- Files: place near parent directory; keep related files close.

Large files:

- Split across groups using indirect blocks (~4MB chunks per group).

## Ext4 Enhancements

Pre-allocation: Reserve extra blocks to avoid fragmentation.

Delayed allocation: Decide placement when writing to disk.

Flexible block groups: Merge several groups to store large files contiguously.
