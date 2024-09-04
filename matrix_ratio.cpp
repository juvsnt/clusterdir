#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 4;
    int matrix[N][N] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    int local_sum = 0;
    int local_diag_sum = 0;

    // Distribute the work among processes
    for (int i = rank; i < N; i += size) {
        for (int j = 0; j < N; ++j) {
            local_sum += matrix[i][j];
            if (i == j) {
                local_diag_sum += matrix[i][j];
            }
        }
    }

    int global_sum = 0;
    int global_diag_sum = 0;

    // Reduce the local sums to global sums
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_diag_sum, &global_diag_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double ratio = static_cast<double>(global_diag_sum) / global_sum;
        std::cout << "Razão da soma da diagonal principal pela soma total: " << ratio << std::endl;
    }

    MPI_Finalize();
    return 0;
}
