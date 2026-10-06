# PART C - MPI Distributed Cluster & Matrix Multiplication

## 1. MPI Cluster Setup

The MPI experiment uses one Master VM and three Worker VMs.

- Master VM → Rank 0
- Worker1 → Rank 1
- Worker2 → Rank 2
- Worker3 → Rank 3

The Master VM coordinates the MPI execution, while the Worker VMs participate in the computation.

## 2. Network Connectivity

Network connectivity between the Master VM and all Worker VMs was verified using the `ping` command.

The connectivity test was successful with **0% packet loss** for all three Worker VMs.

The successful connectivity confirmed that the Master and Worker VMs were able to communicate over the configured network.

<img width="752" height="771" alt="image" src="https://github.com/user-attachments/assets/ac2f6552-af7e-41a3-8e9e-e5c75437a831" />

<img width="764" height="696" alt="image" src="https://github.com/user-attachments/assets/ea7b53d3-f660-474d-99a0-3c23858ed163" />

## 3. SSH Configuration

OpenSSH was configured on the Worker VMs to allow remote access from the Master VM.

The SSH service was verified to be active and running on Worker1, Worker2, and Worker3.

The SSH configuration enables the Master VM to remotely launch and manage processes on the Worker VMs.

<img width="654" height="518" alt="image" src="https://github.com/user-attachments/assets/0bfb2206-f355-4738-8e6f-028e52b72c59" />

<img width="738" height="774" alt="image" src="https://github.com/user-attachments/assets/416b5c36-b570-45b4-b8f0-b986dc202886" />

<img width="775" height="668" alt="image" src="https://github.com/user-attachments/assets/2ab80e36-ce9e-47b5-b093-baa5a53a4304" />

## 4. SSH Communication Test

The Master VM successfully connected to Worker1 and Worker2 using SSH.

The `hostname` command was used to verify that the connection was established with the correct Worker VM.

The successful hostname verification confirmed that the Master VM was communicating with the intended Worker Nodes.

<img width="799" height="721" alt="image" src="https://github.com/user-attachments/assets/43f8240f-812c-4ecf-9242-a1c547ec7e5e" />

<img width="751" height="834" alt="image" src="https://github.com/user-attachments/assets/a51e1545-fe7f-48c3-a022-9a5245afca88" />

<img width="785" height="703" alt="image" src="https://github.com/user-attachments/assets/c46e5eae-7fec-4b02-b8b6-3b6719262bad" />

## 5. MPI Compilation

The MPI matrix multiplication program was compiled using `mpicc`.

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

The executable was copied to Worker1, Worker2, and Worker3 using scp.

```bash
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi
```

The MPI program was then executed using four processes:

```bash
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

## 6. MPI Process Execution and Communication

The MPI program successfully launched four processes across the Master and Worker VMs.

The MPI ranks communicated successfully during execution, confirming that the distributed MPI setup was working correctly.

MPI Execution Output:

<img width="867" height="166" alt="MPI_Execution" src="https://github.com/user-attachments/assets/fe642cb6-7842-4dce-9463-9a6a3f94881c" />

## 7. MPI Working

The matrix multiplication workload was distributed among the four MPI processes.

The Master process coordinated the execution, while the Worker processes performed their assigned computation.

## 8. MPI Result

The MPI matrix multiplication completed successfully in 92.979510 seconds.

The verification value C[0][0] = 4000.00 confirmed the correctness of the result.

The MPI implementation achieved a 2.63× speedup with 65.8% parallel efficiency.

---

