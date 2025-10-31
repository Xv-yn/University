# Security Goals

Operating System (OS) security revolves around three key goals:

- Confidentiality: Prevent unauthorized access to information.
- Integrity: Ensure data and system components remain unaltered by attackers.
- Availability: Guarantee that information and services remain usable and accessible.
- Additional goals: Controlled sharing and non-repudiation.

## Policies and Mechanisms

Security Policies: Define what should be protected and who has access.

Security Mechanisms: Define how the protection is enforced.

Separation of policy and mechanism is essential for flexibility — mechanisms can
remain constant even as policies change

## Securing the Operating System

Privilege Levels

- OS has complete control over hardware (CPU, memory, devices).
- Modern CPUs support multiple privilege levels:
  - Intel: 4 levels (user = level 3, kernel = level 0).
  - ARM: EL0 (user), EL1 (kernel), EL2 (hypervisor), EL3 (secure monitor).
- Memory is divided into user space and kernel space, with system calls used to
  safely transition between them.

## Authentication & Authorization

OS must verify who requests a service before granting access.

Identity-based decisions: Each process and user has unique identifiers used for access
control.

Credentials: OS stores access permissions in a process’s PCB (Process Control Block)
after verification.

## User Authentication

Three Classical Methods

What you know: Passwords, PINs, security questions.

What you have: Tokens, smart cards, physical keys.

What you are: Biometrics (fingerprint, retina, facial recognition, voice, typing
rhythm).

## Password Security

Password Handling

- OS stores hashed passwords, not plaintext.
- Verification: Hash user input → compare to stored hash.
- Uses cryptographic hash functions to make reverse-engineering infeasible.

Vulnerabilities

- Weak, guessable passwords → dictionary attacks.
- Salt Technique
- A random salt is added to each password before hashing:
- Prevents same passwords from producing identical hashes.
- Stored alongside hash in `/etc/shadow` (Linux) — accessible only by root.

## Linux User IDs

Real User ID (RUID): Actual user ID.

Effective User ID (EUID): Temporarily elevated privileges.

Saved User ID (SUID): Allows switching back to a lower privilege level after
privileged work.

## SetUID Bit

Allows programs to temporarily execute with owner’s privileges.

Example: /usr/bin/passwd uses SetUID to let users change their passwords.

Risk: Poorly written SetUID programs can cause privilege escalation.

## Access Control

> [!note] Definition
> Determines who can access what, how, and under what conditions.

Elements

- Subject: Entity requesting access (user/process).
- Object: Resource being accessed (file/device).
- Access Mode: Type of access (read, write, execute).

### Access Control Policies

| **Policy Type**                        | **Description**                                           |
| -------------------------------------- | --------------------------------------------------------- |
| **DAC** (Discretionary Access Control) | Access based on user identity and access rules.           |
| **MAC** (Mandatory Access Control)     | Access based on comparing security labels and clearances. |
| **RBAC** (Role-Based Access Control)   | Access based on roles assigned to users.                  |

## Access Matrix & Lists

Access Matrix Model

- Rows = Subjects (users/processes).
- Columns = Objects (files/devices).
- Cells = Access rights (read/write/execute).

Access Control List (ACL)

- Derived from columns of the matrix.
- Each object has a list of subjects and their permissions — efficient for file
  systems.

Capability List

- Derived from rows of the matrix.
- Each subject has a list of objects they can access — efficient for process-level
  control.

## Linux File Access Control

Basic Structure

- Uses Access Control Lists (ACLs) stored in file metadata (inode/FCB).
- When a file is opened, Linux verifies permissions and creates an internal capability
  object attached to the process’s PCB.
- This speeds up subsequent access checks.

File Permissions

- Each file has 12 protection bits:
  - 9 bits = Read/Write/Execute for owner, group, others.
  - 3 special bits =
    - SetUID: Temporarily elevate privileges.
    - SetGID: Inherit group ownership for new files.
    - Sticky bit: Only the owner can modify/delete the file

### POSIX ACLs

Extends standard permissions to allow specific user or group control:

```txt
user::rwx
user:alice:rw-
group::r--
mask::rwx
other::---
```

Enables finer control than traditional Unix permissions.

## Android Access Control Model

Based on Linux kernel, but adapted for mobile apps.

Problem: Apps come from many developers → potential for malicious access.

Solution:

- Each app runs under its own user ID (principle of least privilege).
- Apps declare required permissions at install time (e.g., camera, contacts).
- Android uses permission labels — like capabilities — stored at installation.
- Labels act as mandatory access control, restricting app actions based on security
  clearance.
- Issue: Users often grant permissions without understanding their implications.

## Role-Based Access Control (RBAC)

Access determined by user’s role, not identity.

Roles correspond to job functions (e.g., Admin, Staff, Student).

Users ↔ Roles ↔ Resources relationship is many-to-many .

Implements Principle of Least Privilege: Each role has only the access necessary for
its tasks.
