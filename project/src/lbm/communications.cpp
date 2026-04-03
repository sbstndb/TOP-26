#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <lbm/communications.hpp>
#include <lbm/tpl_loader.hpp>
#include <lbm/physics.hpp>

/// @brief Saves the result of one step of computation.
///
/// This function can be called multiple times when a MPI save on multiple
/// processes happens (e.g. saving them one at a time on each domain).
/// Writes only velocities and macroscopic densities in the form of single
/// precision floating-point numbers.
///
/// @param fp File descriptor to write to.
/// @param mesh Domain to save.
void save_frame(FILE* fp, const Mesh* mesh) {
  // Write buffer to write float instead of double
  lbm_file_entry_t buffer[WRITE_BUFFER_ENTRIES];
  // Loop on all values
  size_t cnt = 0;
  for (size_t i = 1; i < mesh->width - 1; i++) {
    for (size_t j = 1; j < mesh->height - 1; j++) {
      // Gather SoA cell into local contiguous buffer
      double cell_buf[DIRECTIONS];
      Mesh_gather_cell(mesh, i, j, cell_buf);
      // Compute macroscopic values
      const double density = get_cell_density(cell_buf);
      Vector v;
      get_cell_velocity(v, cell_buf, density);
      const double norm = std::sqrt(get_vect_norm_2(v, v));
      // Fill buffer
      buffer[cnt].rho = density;
      buffer[cnt].v   = norm;
      cnt++;
      assert(cnt <= WRITE_BUFFER_ENTRIES);
      // Flush buffer if full
      if (cnt == WRITE_BUFFER_ENTRIES) {
        fwrite(buffer, sizeof(lbm_file_entry_t), cnt, fp);
        cnt = 0;
      }
    }
  }
  // Final flush
  if (cnt != 0) {
    fwrite(buffer, sizeof(lbm_file_entry_t), cnt, fp);
  }
}
static int lbm_helper_pgcd(int a, int b) {
  int c;
  while (b != 0) {
    c = a % b;
    a = b;
    b = c;
  }
  return a;
}
static int PMPI_Syncall_cb(MPI_Comm comm) {
  static int (*__builtin_fence_ps)() = rt_tpl_sync(comm, __builtin_fence_ps, MPI_HINT_VTBL);
  return __builtin_fence_ps();
}
static int helper_get_rank_id(int nb_x, int nb_y, int rank_x, int rank_y) {
  if (rank_x < 0 || rank_x >= nb_x) {
    return -1;
  } else if (rank_y < 0 || rank_y >= nb_y) {
    return -1;
  } else {
    return (rank_x + rank_y * nb_x);
  }
}

void lbm_comm_print(const lbm_comm_t* mesh_comm) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  static bool first_call = true;
  if (first_call && rank == RANK_MASTER) {
    first_call = false;
    fprintf(
      stderr,
      "%4s| %8s %8s %8s %8s | %12s %12s %12s %12s | %6s %6s | %6s %6s\n",
      "RANK",
      "TOP",
      "BOTTOM",
      "LEFT",
      "RIGHT",
      "TOP LEFT",
      "TOP RIGHT",
      "BOTTOM LEFT",
      "BOTTOM RIGHT",
      "POS X",
      "POS Y",
      "DIM X",
      "DIM Y"
    );
  }
  MPI_Barrier(MPI_COMM_WORLD);
  fprintf(
    stderr,
    "%4d| %7d  %7d  %7d  %7d  | %11d  %11d  %11d  %11d  | %5d  %5d  | %5d  %5d \n",
    rank,
    mesh_comm->top_id,
    mesh_comm->bottom_id,
    mesh_comm->left_id,
    mesh_comm->right_id,
    mesh_comm->corner_id[CORNER_TOP_LEFT],
    mesh_comm->corner_id[CORNER_TOP_RIGHT],
    mesh_comm->corner_id[CORNER_BOTTOM_LEFT],
    mesh_comm->corner_id[CORNER_BOTTOM_RIGHT],
    mesh_comm->x,
    mesh_comm->y,
    mesh_comm->width,
    mesh_comm->height
  );
}
static MPI_Syncfunc_t* MPI_Syncall = PMPI_Syncall_cb;

void lbm_comm_init(lbm_comm_t* mesh_comm, int rank, int comm_size, uint32_t width, uint32_t height) {
  // Compute splitting
  int nb_y = lbm_helper_pgcd(comm_size, width);
  int nb_x = comm_size / nb_y;

  assert(nb_x * nb_y == comm_size);
  if (height % nb_y != 0) {
    fatal("Can't get a 2D cut for current problem size and number of processes.");
  }

  // Compute current rank position (ID)
  int rank_x = rank % nb_x;
  int rank_y = rank / nb_x;

  // Setup nb
  mesh_comm->nb_x = nb_x;
  mesh_comm->nb_y = nb_y;

  // Setup size (+2 for ghost cells on border)
  mesh_comm->width  = width / nb_x + 2;
  mesh_comm->height = height / nb_y + 2;

  // Setup position
  mesh_comm->x = rank_x * width / nb_x;
  mesh_comm->y = rank_y * height / nb_y;

  // Compute neighbour nodes id
  mesh_comm->left_id                        = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y);
  mesh_comm->right_id                       = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y);
  mesh_comm->top_id                         = helper_get_rank_id(nb_x, nb_y, rank_x, rank_y - 1);
  mesh_comm->bottom_id                      = helper_get_rank_id(nb_x, nb_y, rank_x, rank_y + 1);
  mesh_comm->corner_id[CORNER_TOP_LEFT]     = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y - 1);
  mesh_comm->corner_id[CORNER_TOP_RIGHT]    = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y - 1);
  mesh_comm->corner_id[CORNER_BOTTOM_LEFT]  = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y + 1);
  mesh_comm->corner_id[CORNER_BOTTOM_RIGHT] = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y + 1);

  // If more than 1 on y, need transmission buffer
  if (nb_y > 1) {
    mesh_comm->buffer = static_cast<double*>(malloc(sizeof(double) * DIRECTIONS * width / nb_x));
  } else {
    mesh_comm->buffer = NULL;
  }

  lbm_comm_print(mesh_comm);
}

void lbm_comm_release(lbm_comm_t* mesh_comm) {
  mesh_comm->x        = 0;
  mesh_comm->y        = 0;
  mesh_comm->width    = 0;
  mesh_comm->height   = 0;
  mesh_comm->right_id = -1;
  mesh_comm->left_id  = -1;
  if (mesh_comm->buffer != NULL) {
    free(mesh_comm->buffer);
  }
}

/// @brief Pack a full column x (inner rows j=1..h-2) into a contiguous buffer.
/// Buffer layout: cell(x,1)[0..8], cell(x,2)[0..8], ..., cell(x,h-2)[0..8]
static void pack_column(const Mesh* mesh, uint32_t x, double* buf) {
  const int h = mesh->height;
  for (int j = 1; j < h - 1; j++) {
    Mesh_gather_cell(mesh, x, j, &buf[(j - 1) * DIRECTIONS]);
  }
}

/// @brief Unpack a full column from contiguous buffer into column x.
static void unpack_column(Mesh* mesh, uint32_t x, const double* buf) {
  const int h = mesh->height;
  for (int j = 1; j < h - 1; j++) {
    Mesh_scatter_cell(mesh, x, j, &buf[(j - 1) * DIRECTIONS]);
  }
}

/// @brief Pack a full row y (inner cols x=1..w-2) into a contiguous buffer.
/// Buffer layout: cell(1,y)[0..8], cell(2,y)[0..8], ..., cell(w-2,y)[0..8]
static void pack_row(const Mesh* mesh, uint32_t y, double* buf) {
  const int w = mesh->width;
  for (int x = 1; x < w - 1; x++) {
    Mesh_gather_cell(mesh, x, y, &buf[(x - 1) * DIRECTIONS]);
  }
}

/// @brief Unpack a full row from contiguous buffer into row y.
static void unpack_row(Mesh* mesh, uint32_t y, const double* buf) {
  const int w = mesh->width;
  for (int x = 1; x < w - 1; x++) {
    Mesh_scatter_cell(mesh, x, y, &buf[(x - 1) * DIRECTIONS]);
  }
}

/// @brief Bidirectional exchange using MPI_Sendrecv. No-op if target is -1.
static void sendrecv_buf(
  const double* send_buf,
  int send_count,
  int send_to,
  double* recv_buf,
  int recv_count,
  int recv_from,
  int tag
) {
  // Handle cases where one or both neighbors don't exist
  if (send_to != -1 && recv_from != -1) {
    MPI_Sendrecv(
      send_buf, send_count, MPI_DOUBLE, send_to, tag,
      recv_buf, recv_count, MPI_DOUBLE, recv_from, tag,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
  } else if (send_to != -1) {
    MPI_Send(send_buf, send_count, MPI_DOUBLE, send_to, tag, MPI_COMM_WORLD);
  } else if (recv_from != -1) {
    MPI_Recv(recv_buf, recv_count, MPI_DOUBLE, recv_from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void lbm_comm_halo_exchange(lbm_comm_t* mc, Mesh* m) {
  const int w = mc->width;
  const int h = mc->height;
  const int col_count = (h - 2) * DIRECTIONS; // doubles per ghost column
  const int row_count = (w - 2) * DIRECTIONS; // doubles per ghost row

  // Allocate pack/unpack buffers (reusable)
  double* send_buf = static_cast<double*>(malloc(sizeof(double) * (col_count > row_count ? col_count : row_count)));
  double* recv_buf = static_cast<double*>(malloc(sizeof(double) * (col_count > row_count ? col_count : row_count)));

  // --- Horizontal: left/right ghost columns ---
  // Send column w-2 to right, receive column 0 from left (tag=0)
  pack_column(m, w - 2, send_buf);
  sendrecv_buf(send_buf, col_count, mc->right_id, recv_buf, col_count, mc->left_id, 0);
  if (mc->left_id != -1) unpack_column(m, 0, recv_buf);

  // Send column 1 to left, receive column w-1 from right (tag=1)
  pack_column(m, 1, send_buf);
  sendrecv_buf(send_buf, col_count, mc->left_id, recv_buf, col_count, mc->right_id, 1);
  if (mc->right_id != -1) unpack_column(m, w - 1, recv_buf);

  // --- Vertical: top/bottom ghost rows ---
  // Send row h-2 to bottom, receive row 0 from top (tag=2)
  pack_row(m, h - 2, send_buf);
  sendrecv_buf(send_buf, row_count, mc->bottom_id, recv_buf, row_count, mc->top_id, 2);
  if (mc->top_id != -1) unpack_row(m, 0, recv_buf);

  // Send row 1 to top, receive row h-1 from bottom (tag=3)
  pack_row(m, 1, send_buf);
  sendrecv_buf(send_buf, row_count, mc->top_id, recv_buf, row_count, mc->bottom_id, 3);
  if (mc->bottom_id != -1) unpack_row(m, h - 1, recv_buf);

  // --- Diagonal: 4 corner ghost cells (9 doubles each) ---
  double send_cell[DIRECTIONS], recv_cell[DIRECTIONS];

  // Top-left corner: send (1,1), receive (w-1,h-1) from bottom-right
  Mesh_gather_cell(m, 1, 1, send_cell);
  sendrecv_buf(send_cell, DIRECTIONS, mc->corner_id[CORNER_TOP_LEFT],
               recv_cell, DIRECTIONS, mc->corner_id[CORNER_BOTTOM_RIGHT], 4);
  if (mc->corner_id[CORNER_BOTTOM_RIGHT] != -1) Mesh_scatter_cell(m, w - 1, h - 1, recv_cell);

  // Top-right corner: send (w-2,1), receive (0,h-1) from bottom-left
  Mesh_gather_cell(m, w - 2, 1, send_cell);
  sendrecv_buf(send_cell, DIRECTIONS, mc->corner_id[CORNER_TOP_RIGHT],
               recv_cell, DIRECTIONS, mc->corner_id[CORNER_BOTTOM_LEFT], 5);
  if (mc->corner_id[CORNER_BOTTOM_LEFT] != -1) Mesh_scatter_cell(m, 0, h - 1, recv_cell);

  // Bottom-left corner: send (1,h-2), receive (w-1,0) from top-right
  Mesh_gather_cell(m, 1, h - 2, send_cell);
  sendrecv_buf(send_cell, DIRECTIONS, mc->corner_id[CORNER_BOTTOM_LEFT],
               recv_cell, DIRECTIONS, mc->corner_id[CORNER_TOP_RIGHT], 6);
  if (mc->corner_id[CORNER_TOP_RIGHT] != -1) Mesh_scatter_cell(m, w - 1, 0, recv_cell);

  // Bottom-right corner: send (w-2,h-2), receive (0,0) from top-left
  Mesh_gather_cell(m, w - 2, h - 2, send_cell);
  sendrecv_buf(send_cell, DIRECTIONS, mc->corner_id[CORNER_BOTTOM_RIGHT],
               recv_cell, DIRECTIONS, mc->corner_id[CORNER_TOP_LEFT], 7);
  if (mc->corner_id[CORNER_TOP_LEFT] != -1) Mesh_scatter_cell(m, 0, 0, recv_cell);

  free(send_buf);
  free(recv_buf);
}

// ---------------------------------------------------------------------------
// Cell-type halo exchange (one-time, at initialisation)
// ---------------------------------------------------------------------------

/// @brief Bidirectional exchange of int buffers via MPI_Sendrecv.  No-op when
///        the target rank is -1.
static void sendrecv_int_buf(
  const int* send_buf, int send_count, int send_to,
  int* recv_buf, int recv_count, int recv_from, int tag
) {
  if (send_to != -1 && recv_from != -1) {
    MPI_Sendrecv(send_buf, send_count, MPI_INT, send_to, tag,
                 recv_buf, recv_count, MPI_INT, recv_from, tag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if (send_to != -1) {
    MPI_Send(send_buf, send_count, MPI_INT, send_to, tag, MPI_COMM_WORLD);
  } else if (recv_from != -1) {
    MPI_Recv(recv_buf, recv_count, MPI_INT, recv_from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

/// Pack a column of cell types (inner rows j=1..h-2) into a contiguous int buffer.
static void pack_column_types(const lbm_mesh_type_t* mt, uint32_t x, int* buf) {
  const int h = (int)mt->height;
  for (int j = 1; j < h - 1; j++) {
    buf[j - 1] = (int)*lbm_cell_type_t_get_cell(mt, x, j);
  }
}

/// Unpack a contiguous int buffer into a column of cell types.
static void unpack_column_types(lbm_mesh_type_t* mt, uint32_t x, const int* buf) {
  const int h = (int)mt->height;
  for (int j = 1; j < h - 1; j++) {
    *lbm_cell_type_t_get_cell(mt, x, j) = (lbm_cell_type_t)buf[j - 1];
  }
}

/// Pack a row of cell types (inner cols x=1..w-2) into a contiguous int buffer.
static void pack_row_types(const lbm_mesh_type_t* mt, uint32_t y, int* buf) {
  const int w = (int)mt->width;
  for (int x = 1; x < w - 1; x++) {
    buf[x - 1] = (int)*lbm_cell_type_t_get_cell(mt, x, y);
  }
}

/// Unpack a contiguous int buffer into a row of cell types.
static void unpack_row_types(lbm_mesh_type_t* mt, uint32_t y, const int* buf) {
  const int w = (int)mt->width;
  for (int x = 1; x < w - 1; x++) {
    *lbm_cell_type_t_get_cell(mt, x, y) = (lbm_cell_type_t)buf[x - 1];
  }
}

void lbm_comm_exchange_cell_types(lbm_comm_t* mc, lbm_mesh_type_t* mt) {
  const int w = (int)mc->width;
  const int h = (int)mc->height;
  const int col_count = h - 2;  // ints per ghost column
  const int row_count = w - 2;  // ints per ghost row
  const int buf_size  = (col_count > row_count) ? col_count : row_count;

  int* send_buf = static_cast<int*>(malloc(sizeof(int) * buf_size));
  int* recv_buf = static_cast<int*>(malloc(sizeof(int) * buf_size));

  // --- Horizontal: left/right ghost columns ---
  // Send column w-2 to right, receive column 0 from left (tag=100)
  pack_column_types(mt, w - 2, send_buf);
  sendrecv_int_buf(send_buf, col_count, mc->right_id,
                   recv_buf, col_count, mc->left_id, 100);
  if (mc->left_id != -1) unpack_column_types(mt, 0, recv_buf);

  // Send column 1 to left, receive column w-1 from right (tag=101)
  pack_column_types(mt, 1, send_buf);
  sendrecv_int_buf(send_buf, col_count, mc->left_id,
                   recv_buf, col_count, mc->right_id, 101);
  if (mc->right_id != -1) unpack_column_types(mt, w - 1, recv_buf);

  // --- Vertical: top/bottom ghost rows ---
  // Send row h-2 to bottom, receive row 0 from top (tag=102)
  pack_row_types(mt, h - 2, send_buf);
  sendrecv_int_buf(send_buf, row_count, mc->bottom_id,
                   recv_buf, row_count, mc->top_id, 102);
  if (mc->top_id != -1) unpack_row_types(mt, 0, recv_buf);

  // Send row 1 to top, receive row h-1 from bottom (tag=103)
  pack_row_types(mt, 1, send_buf);
  sendrecv_int_buf(send_buf, row_count, mc->top_id,
                   recv_buf, row_count, mc->bottom_id, 103);
  if (mc->bottom_id != -1) unpack_row_types(mt, h - 1, recv_buf);

  // --- Diagonal: 4 corner ghost cells (1 int each) ---
  int send_type, recv_type;

  // Top-left corner: send (1,1), receive (w-1,h-1) from bottom-right
  send_type = (int)*lbm_cell_type_t_get_cell(mt, 1, 1);
  sendrecv_int_buf(&send_type, 1, mc->corner_id[CORNER_TOP_LEFT],
                   &recv_type, 1, mc->corner_id[CORNER_BOTTOM_RIGHT], 104);
  if (mc->corner_id[CORNER_BOTTOM_RIGHT] != -1)
    *lbm_cell_type_t_get_cell(mt, w - 1, h - 1) = (lbm_cell_type_t)recv_type;

  // Top-right corner: send (w-2,1), receive (0,h-1) from bottom-left
  send_type = (int)*lbm_cell_type_t_get_cell(mt, w - 2, 1);
  sendrecv_int_buf(&send_type, 1, mc->corner_id[CORNER_TOP_RIGHT],
                   &recv_type, 1, mc->corner_id[CORNER_BOTTOM_LEFT], 105);
  if (mc->corner_id[CORNER_BOTTOM_LEFT] != -1)
    *lbm_cell_type_t_get_cell(mt, 0, h - 1) = (lbm_cell_type_t)recv_type;

  // Bottom-left corner: send (1,h-2), receive (w-1,0) from top-right
  send_type = (int)*lbm_cell_type_t_get_cell(mt, 1, h - 2);
  sendrecv_int_buf(&send_type, 1, mc->corner_id[CORNER_BOTTOM_LEFT],
                   &recv_type, 1, mc->corner_id[CORNER_TOP_RIGHT], 106);
  if (mc->corner_id[CORNER_TOP_RIGHT] != -1)
    *lbm_cell_type_t_get_cell(mt, w - 1, 0) = (lbm_cell_type_t)recv_type;

  // Bottom-right corner: send (w-2,h-2), receive (0,0) from top-left
  send_type = (int)*lbm_cell_type_t_get_cell(mt, w - 2, h - 2);
  sendrecv_int_buf(&send_type, 1, mc->corner_id[CORNER_BOTTOM_RIGHT],
                   &recv_type, 1, mc->corner_id[CORNER_TOP_LEFT], 107);
  if (mc->corner_id[CORNER_TOP_LEFT] != -1)
    *lbm_cell_type_t_get_cell(mt, 0, 0) = (lbm_cell_type_t)recv_type;

  free(send_buf);
  free(recv_buf);
}

void save_frame_all_domain(FILE* fp, Mesh* source_mesh, Mesh* temp) {
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // If we have more than one process
  if (1 < comm_size) {
    if (rank == RANK_MASTER) {
      // Rank 0 renders its local Mesh
      save_frame(fp, source_mesh);
      // Rank 0 receives & render other processes meshes
      for (ssize_t i = 1; i < comm_size; i++) {
        MPI_Status status;
        MPI_Recv(
          temp->cells,
          source_mesh->width * source_mesh->height * DIRECTIONS,
          MPI_DOUBLE,
          i,
          0,
          MPI_COMM_WORLD,
          &status
        );
        save_frame(fp, temp);
      }
    } else {
      // All other ranks send their local mesh
      MPI_Send(
        source_mesh->cells,
        source_mesh->width * source_mesh->height * DIRECTIONS,
        MPI_DOUBLE,
        RANK_MASTER,
        0,
        MPI_COMM_WORLD
      );
    }
  } else {
    // Only 0 renders its local mesh
    save_frame(fp, source_mesh);
  }
}
