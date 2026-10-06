# Comparative Performance Study of Matrix Multiplication

> **High Performance Computing: Sequential CPU vs OpenMP vs MPI vs CUDA**  
> Experimental comparison of matrix multiplication using sequential processing, shared-memory parallelism, distributed-memory computing, and GPU acceleration.


---

## 1. Overview

Matrix multiplication is an important computational operation used in scientific computing, machine learning, simulations, image processing, and many other applications.

This project compares four different approaches for multiplying two dense matrices:

- Sequential CPU execution
- OpenMP shared-memory parallel execution
- MPI distributed-memory execution
- CUDA GPU-based execution

All implementations operate on matrices of size **4000 × 4000**.

The main purpose of the experiment is to compare the execution time and performance improvement obtained using different parallel computing architectures.

---

## 2. Problem Definition

Two square matrices `A` and `B` are multiplied to obtain matrix `C`.

The mathematical operation is:

$$
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \times B[k][j]
$$

where:

- `N = 4000`
- `A` is a `4000 × 4000` matrix
- `B` is a `4000 × 4000` matrix
- `C` is the resulting `4000 × 4000` matrix

Each element of matrices `A` and `B` is initialized to `1.0`.

Therefore:

$$
C[i][j] = 4000
$$

The verification condition is:


```text
C[0][0] = 4000.00
```

---

## 3. Computational Requirements
### Number of Floating-Point Operations

Matrix multiplication requires approximately:

$$ 2N^3 $$

operations.

For N = 4000:

$$ 2(4000)^3 = 128,000,000,000 $$

Therefore, the computation represents approximately:

```
128 GFLOPs
```

Memory Requirement

Each matrix contains:

```
4000 × 4000 = 16,000,000 elements
```

Using double-precision values:

```
1 element = 8 bytes
```

Memory required for one matrix:

```
16,000,000 × 8 = 128 MB
```

Three matrices require approximately:

```
A + B + C = 384 MB
```

---

## 4. Experimental Results

The following table summarizes the performance obtained from all four implementations.

| Implementation | Computing Architecture | Execution Time | Speedup | Efficiency | GFLOPs | Verification |
|---|---|---:|---:|---:|---:|---|
| Sequential CPU | 1 CPU Core | **247.290475 s** | **1.00×** | **100%** | **0.524** | PASS |
| MPI | 4 Distributed VMs | **92.979510 s** | **2.63×** | **65.8%** | **1.377** | PASS |
| OpenMP | 8 CPU Threads | **40.545825 s** | **6.02×** | **75.3%** | **3.157** | PASS |
| CUDA - Total | NVIDIA GPU | **0.343020 s** | **711.68×** | — | **373.156** | PASS |
| CUDA - Kernel | NVIDIA GPU | **0.316872 s** | **770.40×** | — | **403.948** | PASS |

---

## 5. Sequential Matrix Multiplication

### 5.1 Description

The sequential implementation performs matrix multiplication using a single CPU execution thread.

Three nested loops are used:

```bash
for each row i
    for each column j
        for each element k
            C[i][j] += A[i][k] × B[k][j]
```

Since there is no parallel processing, the complete computation is performed by one CPU core.

The execution time obtained from this implementation is used as the baseline for calculating speedup.

### 5.2 Environment Preparation

The experiment was performed using Ubuntu through WSL2.

The Linux environment was verified before compiling the program.

GCC was also checked to ensure that the C compiler was available.

### 5.3 Compilation and Execution

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential
```

## 5.4 Output

Screenshot / Image:

<img width="880" height="252" alt="sequential_olp" src="https://github.com/user-attachments/assets/4f112442-c2f8-46e4-a04a-1c5d3599150a" />

## 5.5 Result

The sequential implementation completed the matrix multiplication successfully.

The measured execution time was:

```
247.290475 seconds
```

The verification value was:

```
C[0][0] = 4000.00
```

This confirms that the matrix multiplication produced the expected result.

The sequential implementation is used as the reference implementation for calculating the speedup of OpenMP, MPI, and CUDA.

---

## 6. MPI Distributed Matrix Multiplication

### 6.1 MPI Cluster Configuration

The MPI implementation uses four Ubuntu virtual machines.

| Machine | MPI Rank | Role |
|---|---:|---|
| Master VM | Rank 0 | Coordinator |
| Worker 1 | Rank 1 | Computation |
| Worker 2 | Rank 2 | Computation |
| Worker 3 | Rank 3 | Computation |

The Master VM starts the MPI program and coordinates the participating worker processes.

### 6.2 Network Connectivity

Before executing the MPI program, network connectivity between the Master VM and Worker VMs was tested using ping.

All required worker connections were successfully verified with:

```
0% packet loss
```

This confirmed that the virtual machines were able to communicate through the configured network.

Screenshot / Image:

<img width="752" height="771" alt="image" src="https://github.com/user-attachments/assets/ac2f6552-af7e-41a3-8e9e-e5c75437a831" />

<img width="764" height="696" alt="image" src="https://github.com/user-attachments/assets/ea7b53d3-f660-474d-99a0-3c23858ed163" />

### 6.3 SSH Configuration

SSH was configured on the worker machines so that the Master VM could remotely access them.

The SSH service was checked and confirmed to be active on the worker VMs.

SSH allows the Master VM to launch MPI processes on the worker machines.

Screenshot / Image:

<img width="654" height="518" alt="image" src="https://github.com/user-attachments/assets/0bfb2206-f355-4738-8e6f-028e52b72c59" />

<img width="738" height="774" alt="image" src="https://github.com/user-attachments/assets/416b5c36-b570-45b4-b8f0-b986dc202886" />

<img width="775" height="668" alt="image" src="https://github.com/user-attachments/assets/2ab80e36-ce9e-47b5-b093-baa5a53a4304" />

### 6.4 SSH Communication Verification

The Master VM was tested against the worker machines using SSH.

The hostname command was used after connecting to verify that the correct worker machine was reached.

Example:

```bash
ssh worker1
hostname
```

The hostname of the corresponding worker was displayed successfully.

Screenshot / Image:

<img width="799" height="721" alt="image" src="https://github.com/user-attachments/assets/43f8240f-812c-4ecf-9242-a1c547ec7e5e" />

<img width="751" height="834" alt="image" src="https://github.com/user-attachments/assets/a51e1545-fe7f-48c3-a022-9a5245afca88" />

<img width="785" height="703" alt="image" src="https://github.com/user-attachments/assets/c46e5eae-7fec-4b02-b8b6-3b6719262bad" />

### 6.5 MPI Compilation

The MPI program was compiled using:

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

The generated executable was copied to the worker machines.

```bash
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi
```

The MPI program was then executed using four processes:

```
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

### 6.6 MPI Process Execution and Communication

The MPI program was executed using four processes across the Master VM and three Worker VMs.

The output confirms that:

- Rank 0 is running on the Master VM.
- Rank 1 is running on Worker 1.
- Rank 2 is running on Worker 2.
- Rank 3 is running on Worker 3.
- Rank 0 successfully sent data to Rank 1.
- Rank 1 successfully received the data from Rank 0.

This confirms that the MPI processes were successfully launched across the configured virtual machines and that communication between MPI ranks was working correctly.

**MPI Execution Output:**

<img width="867" height="166" alt="Screenshot 2026-10-06 234522" src="https://github.com/user-attachments/assets/9a2ac58a-7f39-431c-a1dd-654116cf8bbf" />

### 6.7 MPI Compilation

The MPI program was compiled using:

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

The generated executable was copied to the worker machines.

```bash
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi
```

The MPI program was then executed using four processes:

```bash
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

### 6.8 MPI Working

The MPI implementation divides the matrix computation among multiple processes.

The Master process coordinates the execution, while the worker processes perform portions of the matrix multiplication.

The computation is distributed across the four virtual machines.

### 6.9 MPI Result

The measured execution time was:

```
92.979510 seconds
```

The verification result was:

```
C[0][0] = 4000.00
```

Therefore, the MPI implementation produced the correct matrix multiplication result.

Compared with the sequential execution, MPI achieved approximately:

```
2.63× speedup
```

The calculated parallel efficiency was:

```
65.8%
```

## 7. OpenMP Shared-Memory Matrix Multiplication

### 7.1 Description

OpenMP was used to parallelize matrix multiplication using multiple CPU threads.

The experiment was performed using:

```
8 OpenMP threads
```

Unlike MPI, OpenMP uses a shared-memory architecture where all threads operate within the same system memory.

### 7.2 OpenMP Setup

OpenMP was configured in the Ubuntu environment.

The number of OpenMP threads was specified using:

```bash
export OMP_NUM_THREADS=8
```

### 7.3 Compilation

The OpenMP program was compiled using:

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

The program was executed using:

```bash
./matrix_openmp
```

### 7.4 OpenMP Output

Screenshot / Image:

<img width="762" height="294" alt="openmp" src="https://github.com/user-attachments/assets/553569cd-33ad-4bb8-90e2-89ffd1049572" />

### 7.5 OpenMP Result

The OpenMP implementation completed the multiplication successfully.

Execution time:

```
40.545825 seconds
```

Verification:

```
C[0][0] = 4000.00
```

The measured speedup relative to the sequential implementation was:

```
6.02×
```

The parallel efficiency was:

```
75.3%
```

OpenMP therefore provided considerably better performance than both the sequential implementation and the MPI experiment in this setup.

---

## 8. CUDA GPU Matrix Multiplication

### 8.1 CUDA Approach

The CUDA implementation moves the matrix multiplication workload from the CPU to an NVIDIA GPU.

Each output element of matrix C can be calculated independently, allowing a very large number of GPU threads to execute the computation concurrently.

The experiment uses:

```
4000 × 4000 matrices
```

and a two-dimensional CUDA grid.

The configuration consists of:

```
250 × 250 blocks
```

with:

```
16 × 16 threads per block
```

Therefore:

```
16,000,000 GPU threads
```

are launched across the complete grid.

### 8.2 CUDA Compilation

The CUDA program can be compiled using:

```bash
nvcc -O2 matrix_cuda.cu -o matrix_cuda.exe
```

The executable can then be run using:

```bash
./matrix_cuda.exe
```

### 8.3 CUDA Output

Generated CUDA Output Image:

<img width="681" height="243" alt="Screenshot 2026-10-06 204743" src="https://github.com/user-attachments/assets/747ddbdf-3df5-4e48-8ee9-26c8fb8d8902" />

### 8.4 CUDA Performance

The CUDA experiment produced two important timing measurements.

#### GPU Kernel Time

```
0.316872 seconds
```

The corresponding speedup was:

```
770.40×
```

The calculated GPU kernel throughput was:

```
403.948 GFLOPs
```

#### Complete Host-Device Execution

Including host-to-device and device-to-host data transfers:

```
0.343020 seconds
```

The overall speedup was:

```
711.68×
```

The complete throughput was:

```
373.156 GFLOPs
```

### 8.5 CUDA Verification

The GPU result was verified using:

```
C[0][0] = 4000.00
```

and:

```
C[3999][3999] = 4000.00
```

The verification was successful.

```
Verification: PASS
```

---

## 9. Performance Comparison

The sequential implementation is used as the baseline:

```
Sequential Time = 244.120000 seconds
```

Speedup is calculated using:

$$ Speedup = \frac{T_{Sequential}}{T_{Parallel}} $$

### 9.1 Speedup Comparison

| Method | Execution Time | Speedup |
|---|---:|---:|
| Sequential | 247.290475 s | 1.00× |
| MPI | 92.979510 s | 2.63× |
| OpenMP | 40.545825 s | 6.02× |
| CUDA Total | 0.343020 s | 711.68× |
| CUDA Kernel | 0.316872 s | 770.40× |

### 9.2 Speedup Graph

<img width="728" height="512" alt="Screenshot 2026-10-06 192645" src="https://github.com/user-attachments/assets/064560cc-d0f7-4c85-97ae-303368c91494" />

---

## 10. Execution Time Comparison

Execution time is another important measure of performance.

A lower execution time indicates faster completion of the matrix multiplication.

| Implementation | Execution Time |
|---|---:|
| Sequential | 247.290475 s |
| MPI | 92.979510 s |
| OpenMP | 40.545825 s |
| CUDA | 0.343020 s |

### 10.1 Execution Time Graph

<img width="744" height="533" alt="Screenshot 2026-10-06 194134" src="https://github.com/user-attachments/assets/b1844885-d42c-4334-8733-3903636201d3" />

---

## 11. Performance Analysis

The experimental results show significant differences between the four computing approaches.

### Sequential CPU

The sequential implementation required:

```
247.290475 seconds
```

Since only one CPU execution thread performs the computation, it provides the lowest computational parallelism.

It is therefore used as the baseline.

### MPI

MPI reduced the execution time to:

```
92.979510 seconds
```

The distributed implementation achieved:

```
2.63× speedup
```

However, communication between virtual machines introduces additional overhead.

### OpenMP

OpenMP achieved a much lower execution time:

```
40.545825 seconds
```

with:

```
6.02× speedup
```

The shared-memory architecture allows multiple CPU threads to access the same memory space without the network communication required by MPI.

### CUDA

CUDA achieved the highest performance.

The complete GPU execution time was:

```
0.343020 seconds
```

resulting in:

```
711.68× speedup
```

The kernel-only execution time was even lower:

```
0.316872 seconds
```

which corresponds to:

```
770.40× speedup
```

The GPU is able to execute a very large number of threads concurrently, making it highly suitable for this type of data-parallel computation.

---

## 12. Overall Comparison

| Feature | Sequential | MPI | OpenMP | CUDA |
|---|---|---|---|---|
| Processing Type | Serial | Distributed | Shared Memory | GPU Parallel |
| Processing Units | 1 Core | 4 VMs | 8 Threads | GPU Threads |
| Memory Model | Single Memory | Distributed | Shared | GPU Global/Device Memory |
| Communication | None | Network | Shared Memory | Host-GPU |
| Execution Time | 247.290 s | 92.980 s | 40.546 s | 0.343 s |
| Speedup | 1.00× | 2.63× | 6.02× | 711.68× |
| Verification | PASS | PASS | PASS | PASS |

---

## 13. Key Observations

The experiment produced the following observations:

- The sequential implementation provides the baseline performance.
- MPI improves performance by distributing computation across multiple virtual machines.
- MPI performance is affected by communication and network overhead between the VMs.
- OpenMP provides better performance than MPI in this experimental setup because all threads use shared memory.
- CUDA provides the largest performance improvement because matrix multiplication contains a high degree of data parallelism.
- The CUDA implementation achieved more than 700× speedup compared with the sequential CPU implementation.
- All four implementations successfully produced:

```
C[0][0] = 4000.00
```

which confirms correctness.

---

## 14. Technologies Used

The following technologies were used for the experiments:

- C
- GCC
- Ubuntu
- WSL2
- OpenMP
- MPI
- OpenSSH
- NVIDIA CUDA
- nvcc

---

## 15. Project Structure

# Project Structure

```text
Parallel-and-GPU-Computing/
│
├── README.md
├── Sequential.md
├── OpenMP.md
├── MPI.md
└── CUDA.md
```

---

## 16. Conclusion

This experiment compared sequential, MPI, OpenMP, and CUDA implementations of dense matrix multiplication using 4000 × 4000 matrices.

The sequential implementation required 247.290475 seconds and was used as the baseline.

The MPI implementation reduced the execution time to 92.979510 seconds, achieving a 2.63× speedup.

The OpenMP implementation performed better, completing the calculation in 40.545825 seconds and achieving a 6.02× speedup.

The CUDA implementation provided the largest improvement, completing the complete host-device execution in only 0.343020 seconds and achieving a 711.68× speedup.

The CUDA kernel itself required only 0.316872 seconds, corresponding to a 770.40× speedup.

Overall, the results demonstrate that GPU acceleration can provide extremely high performance for highly parallel numerical workloads such as dense matrix multiplication.

---
