//ChenGao 70049738
//EECS 587 HW3

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <sstream>
#include <utility>
#include <mpi.h>

using std::pair;
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::tie;
using std::stoi;
using std::ostream;
using std::ostringstream;
using std::max;
using std::min;

int rank;

double f(double x)
{
    double y = x;
    for (auto i = 1; i <= 10; ++i) {
        y += sin(i * x) / (1 << i);
    }
    return y;
}

// Initialize A function
double initCell(int i, int j) // i & j are global index
{
    return i * sin(i) + j * cos(j) + sqrt(i + j);
}


class CellProcessor
{
    using Index = pair<int, int>;
    using Dimension = pair<int, int>;

    // The global matrix info, m & n
    Dimension matrix_shape;

    // My location
    int sp;          //sqrt(p)
    int rank;        //my processor rank
    Index cpu_idx;   //how many rows of procs above me, and how many cols of procs before me

    // Local cell storage
    Dimension cell_shape;    // size of submatrix in each processor
    Dimension actual_shape;  // size of submatrix in each processor (include ghost)
    Index ghost_offset;      // transfer between local index and local index with ghost
    Index global_offset;     // transfer between local index and global index. gi = li + global_offset
    double *cell_data;       // array to record original data
    double *cell_f;          // array to record f data

    enum NeighborBits
    {
        Left   = 1 << 0,  //Left   = 0001
        Right  = 1 << 1,  //Right  = 0010
        Top    = 1 << 2,  //Top    = 0100
        Bottom = 1 << 3,  //Bottom = 1000
    };
    uint8_t neighbors;

    // MPI related
    MPI_Request requests[8];
    MPI_Datatype mpi_col_type;
    MPI_Datatype mpi_row_type;


public:
    CellProcessor(int m, int n, int sp, int rank)
        : matrix_shape(m, n)
        , sp(sp)
        , rank(rank)
        , cell_data(nullptr)
        , cell_f(nullptr)
    {
        // calc my location (row, col) in rank, form a square grid
        // in sp = 3 example:
        // (0, 0) (0, 1) (0, 2)
        // (1, 0) (1, 1) (1, 2)
        // (2, 0) (2, 1) (2, 2)
        cpu_idx = {rank / sp, rank % sp};

        // detect neighbor
        detectNeighbor();

        // allocate cell
        allocateLocalCell();
    }

    ~CellProcessor()
    {
        delete[] cell_data;
        delete[] cell_f;
    }

    void fillInInitialValue()
    {
        int gi, gj;
        for (auto i = 0; i != cell_shape.first; ++i) {
            for (auto j = 0; j!= cell_shape.second; ++j) {
                tie(gi, gj) = localToGlobal({i, j});
                *cellDataAt(i, j) = initCell(gi, gj);
            }
        }
        // also write initial data at global edges to shadow buffer, which won't change over the time
        // so we won't touch them after this
        // Note here we are writing data using read and read from write so that we are essentially
        // write to shadow buffer.
        if (cpu_idx.first == 0) {
            for (auto j = 0; j != cell_shape.second; ++j) {
                *cacheDataAt(0, j) = f(*cellDataAt(0, j));
            }
        }
        if (cpu_idx.first == sp - 1) {
            for (auto j = 0; j != cell_shape.second; ++j) {
                *cacheDataAt(cell_shape.first - 1, j) = f(*cellDataAt(cell_shape.first - 1, j));
            }
        }
        if (cpu_idx.second == 0) {
            for (auto i = 0; i != cell_shape.first; ++i) {
                *cacheDataAt(i, 0) = f(*cellDataAt(i, 0));
            }
        }
        if (cpu_idx.second == sp - 1) {
            for (auto i = 0; i != cell_shape.first; ++i) {
                *cacheDataAt(i, cell_shape.second - 1) = f(*cellDataAt(i, cell_shape.second - 1));
            }
        }
    }

    void doIteration(int nIter)
    {
        while (nIter--) {
            updateFcache();  //calc f(A_prious) matrix 

            syncGhostValue();// fill f(A_prious)'s ghost cell

            updateCell();    // update A using f(A_prious) matrix. This is to avoid calc f muti times.

        }
    }

    pair<double, double> doVerification()
    {
        // first calculate myself
        double sum[2] = {0.0, 0.0};
        for (auto i = 0; i != cell_shape.first; ++i) {
            for (auto j = 0; j != cell_shape.second; ++j) {
                sum[0] += *cellDataAt(i, j);
                sum[1] += *cellDataAt(i, j) * *cellDataAt(i, j);
            }
        }

        // reduce to root
        double recvsum[2];
        MPI_Reduce(sum, recvsum, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        return {recvsum[0], recvsum[1]};
    }

private:
    // Calc if there is a proc in neighborhood
    void detectNeighbor()
    {
        neighbors = 0;

        if (cpu_idx.first > 0) {
            neighbors += Top;
        }
        if (cpu_idx.first < sp - 1) {
            neighbors += Bottom;
        }
        if (cpu_idx.second > 0) {
            neighbors += Left;
        }
        if (cpu_idx.second < sp - 1) {
            neighbors += Right;
        }
    }

    void allocateLocalCell()
    {
        // calc the submatrix I need to handle
        cell_shape = {
            (matrix_shape.first + sp) / sp,
            (matrix_shape.second + sp) / sp
        };

        // calc the global offset first to avoid to calc full cell again. MUST calc here to avoid the case that cell_shape might change
        global_offset = {
            cell_shape.first * cpu_idx.first,
            cell_shape.second * cpu_idx.second
        };

        // adjust submatrix size for last row and last col

        //if last row, adjust row number
        if (cpu_idx.first == sp - 1) { 
            cell_shape.first = matrix_shape.first % cell_shape.first;
        }
        //if last col, adjust col number
        if (cpu_idx.second == sp - 1) {
            cell_shape.second = matrix_shape.second % cell_shape.second;
        }

        // include ghost areas that will be received from other processor. Updata true submatrix size(may contain ghost)
        // also calc ghost_offset. if there is a ghost col at left ot at top, this will affect local index. 
        ghost_offset = {0, 0};
        actual_shape = cell_shape;
        if (neighbors & Top) {
            ghost_offset.first += 1;
            actual_shape.first += 1;
        }
        if (neighbors & Bottom) {
            actual_shape.first += 1;
        }
        if (neighbors & Left) {
            ghost_offset.second += 1;
            actual_shape.second += 1;
        }
        if (neighbors & Right) {
            actual_shape.second += 1;
        }

        // construct new strided column-based double MPI type
        MPI_Type_vector(cell_shape.first, /* number of blocks */
                        1, /* number of elements of old_type in each block */
                        actual_shape.second, /* number of elements of old_type between start of each block */
                        MPI_DOUBLE, /* old_type */
                        &mpi_col_type);
        MPI_Type_commit(&mpi_col_type);


        // construct new continues row-base double MPI type
        MPI_Type_contiguous(cell_shape.second, /* count */
                            MPI_DOUBLE, /* old_type */
                            &mpi_row_type);
        MPI_Type_commit(&mpi_row_type);

        // allocate data
        cell_data = new double[actual_shape.first * actual_shape.second];
        cell_f = new double[actual_shape.first * actual_shape.second];
    }

    double *cellDataAt(int row, int col)
    {
        // should check null pointer, omitted here only for simplicity.
        return cell_data + localToGhost1D(row, col);
    }

    double *cacheDataAt(int row, int col)
    {
        // should check null pointer, omitted here only for simplicity.
        return cell_f + localToGhost1D(row, col);
    }

    // transfer from local index (without ghost) to 1D index
    int localToGhost1D(int row, int col)
    {
        row += ghost_offset.first;
        col += ghost_offset.second;
        return row * actual_shape.second + col;
    }

	// transfer from local index to global index. This is only for initialization
    Index localToGlobal(const Index &local)
    {
        return {global_offset.first + local.first, global_offset.second + local.second};
    }

    void syncGhostValue()
    {
        int count = 0;
        if (neighbors & Top) {
            MPI_Isend(cacheDataAt(0, 0), 1, mpi_row_type, /* data, length, type */
                      rank - sp, /* destination */
                      0, /* tag */
                      MPI_COMM_WORLD,
                      &requests[count++]);

            MPI_Irecv(cacheDataAt(-1, 0), 1, mpi_row_type, /* data, length, type */
                      rank - sp, /* destination */
                      0, /* tag */
                      MPI_COMM_WORLD,
                      &requests[count++]);
        }
        if (neighbors & Bottom) {
            MPI_Isend(cacheDataAt(cell_shape.first - 1, 0), 1, /* data, length */
                      mpi_row_type, /* type */
                      rank + sp, /* destination */
                      0, /* tag */
                      MPI_COMM_WORLD,
                      &requests[count++]);

            MPI_Irecv(cacheDataAt(cell_shape.first, 0), 1, /* data, length */
                      mpi_row_type, /* type */
                      rank + sp, /* destination */
                      0, /* tag */
                      MPI_COMM_WORLD,
                      &requests[count++]);
        }
        if (neighbors & Left) {
            MPI_Isend(cacheDataAt(0, 0), 1, /* data, length */
                      mpi_col_type, /* type */
                      rank - 1, /* destination */
                      0, /* tag */
                      MPI_COMM_WORLD,
                      &requests[count++]);

            MPI_Irecv(cacheDataAt(0, -1), 1, /* data, length */
                      mpi_col_type, /* type */
                      rank - 1, /* destination */
                      0, /* tag */
                      MPI_COMM_WORLD,
                      &requests[count++]);
        }
        if (neighbors & Right) {
            MPI_Isend(cacheDataAt(0, cell_shape.second - 1), 1, /* data, length */
                      mpi_col_type, /* type */
                      rank + 1, /* destination */
                      0, /* tag */
                      MPI_COMM_WORLD,
                      &requests[count++]);

            MPI_Irecv(cacheDataAt(0, cell_shape.second), 1, /* data, length */
                      mpi_col_type, /* type */
                      rank + 1, /* destination */
                      0, /* tag */
                      MPI_COMM_WORLD,
                      &requests[count++]);
        }
        MPI_Waitall(count, requests, MPI_STATUS_IGNORE);
    }

    void updateFcache()
    {
        int startRow = 0;
        int startCol = 0;
        int endRow = cell_shape.first;
        int endCol = cell_shape.second;

        if (cpu_idx.first == 0) {
            startRow += 1;
        }
        if (cpu_idx.first == sp - 1) {
            endRow -= 1;
        }
        if (cpu_idx.second == 0) {
            startCol += 1;
        }
        if (cpu_idx.second == sp - 1) {
            endCol -= 1;
        }

        for (auto i = startRow; i != endRow; ++i) {
            for (auto j = startCol; j != endCol; ++j) {
                *cacheDataAt(i, j) = f(*cellDataAt(i, j));
            }
        }
    }

        void updateCell()
    {
        int startRow = 0;
        int startCol = 0;
        int endRow = cell_shape.first;
        int endCol = cell_shape.second;

        // skip row/col that are global edge
        if (cpu_idx.first == 0) {
            startRow += 1;
        }
        if (cpu_idx.first == sp - 1) {
            endRow -= 1;
        }
        if (cpu_idx.second == 0) {
            startCol += 1;
        }
        if (cpu_idx.second == sp - 1) {
            endCol -= 1;
        }

        // Note: use i < endRow not i != endRow
        // as endRow could be less than startRow with previous adjustment
        // the same for startCol/endCol
        for (auto i = startRow; i < endRow; ++i) {
            for (auto j = startCol; j < endCol; ++j) {
                *cellDataAt(i, j) = calculateCell(i, j);
            }
        }
    }

    //Come up with a speedup. No need to calc f each time but save them in cache
    double calculateCell(int i, int j)
    {
        auto tmp = 
                      *cacheDataAt(i, j)
                    //f(*cellDataAt(i, j))
                    + *cacheDataAt(i-1, j)
                    //+ f(*cellDataAt(i-1, j))
                    + *cacheDataAt(i+1, j)
                    //+ f(*cellDataAt(i+1, j))
                    + *cacheDataAt(i, j-1)
                    //+ f(*cellDataAt(i, j-1))
                    + *cacheDataAt(i, j+1);
                    //+ f(*cellDataAt(i, j+1));
        tmp /= 5;
        return max(-100.0, min(100.0, tmp));
    }
};


int main(int argc, char **argv) {
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Initialize A MPI
    int sp = (int) sqrt(size); // gaurantee that size is squared
    int m = stoi(*(++argv));
    int n = stoi(*(++argv));
    CellProcessor processor(m, n, sp, rank);
    processor.fillInInitialValue();

    // Step 2: Use MPI_BARRIER, then have the root process (process 0) call MPI_WTIME.
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = 0;
    if (rank == 0) {
        t_start = MPI_Wtime();
    }

    // Step 3: Do 10 iterations
    processor.doIteration(10);

    // Step 4: Compute the verification values and send them to the root process.
    double sum, square_sum;
    tie(sum, square_sum) = processor.doVerification();

    if (rank == 0) {
        // Step 5: Stop timer.
        double running_time = MPI_Wtime() - t_start;

        // Step 6: Print out the elapsed time and the verification.
        cout << "Elapsed time: " << running_time << endl;
        cout << "Sum of entries: " << sum << endl
             << "Sum of the square of entries: " << square_sum << endl;
    }

    MPI_Finalize();

    return 0;
}
