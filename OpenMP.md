# Part B - OpenMP Matrix Multiplication

## 1. OpenMP Setup

OpenMP and OpenSSH were set up on the required Ubuntu systems for running the matrix multiplication program.

The configuration allowed the systems to communicate and run the MPI program across multiple processes.

## 2. OpenMP Matrix Multiplication Program

The program uses a matrix size of 4000 × 4000.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    double start, end;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    start = omp_get_wtime();

    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                C[i * N + j] +=
                    A[i * N + k] *
                    B[k * N + j];
            }
        }
    }

    end = omp_get_wtime();

    printf("OpenMP Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Number of Threads Used = %d\n", omp_get_max_threads());
    printf("Execution Time = %f seconds\n", end - start);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
```

## 2. Working and Output

The OpenMP matrix multiplication program was compiled and executed successfully using 8 threads.

The program multiplied two 4000 × 4000 matrices and completed the calculation successfully.

The output was verified to ensure that the matrix multiplication result was correct.

<img width="762" height="294" alt="openmp" src="https://github.com/user-attachments/assets/553569cd-33ad-4bb8-90e2-89ffd1049572" />

## 3. Result

The OpenMP matrix multiplication program was executed successfully.

The program completed the calculation in 40.545825 seconds.

The verification value **C[0][0] = 4000.00** was obtained, confirming that the result was correct.
