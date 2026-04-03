#pragma once

#include <cstdint>

#include <mpi.h>

#include <lbm/structures.hpp>

/// Definition of the master's process ID.
#define RANK_MASTER 0

/// @brief Definition of the different types of cell to know which process to * apply when computing.
typedef enum lbm_corner_pos_e {
  CORNER_TOP_LEFT     = 0,
  CORNER_TOP_RIGHT    = 1,
  CORNER_BOTTOM_LEFT  = 2,
  CORNER_BOTTOM_RIGHT = 3,
} lbm_corner_pos_t;

/// @brief Type of communication.
typedef enum lbm_comm_type_e {
  COMM_SEND,
  COMM_RECV
} lbm_comm_type_t;

/// @brief Structure used to store information about the communications.
typedef struct lbm_comm_t_s {
  /// X position of the local mesh in the global one (origin).
  uint32_t x;
  /// Y position of the local mesh in the global one (origin).
  uint32_t y;
  /// Size of the local mesh.
  uint32_t width;
  uint32_t height;
  int nb_x;
  int nb_y;
  /// ID of the right neighboor, -1 if none.
  int right_id;
  /// ID of the left neighboor, -1 if none.
  int left_id;
  /// ID of the top neighboor, -1 if none.
  int top_id;
  /// ID of the bottom neighboor, -1 if none.
  int bottom_id;
  int corner_id[4];
  /// Async requests for non-blocking halo exchange.
  MPI_Request requests[32];
  int n_requests;
  /// Pre-allocated pack/unpack buffers for halo exchange.
  double* halo_send_bufs[8]; // up to 8 messages (2 horiz + 2 vert + 4 corners)
  double* halo_recv_bufs[8];
  lbm_mesh_cell_t buffer;
} lbm_comm_t;
typedef int MPI_Syncfunc_t(MPI_Comm);

static inline int lbm_comm_width(const lbm_comm_t* mc) {
  return mc->width;
}
static inline int lbm_comm_height(const lbm_comm_t* mc) {
  return mc->height;
}

/// @brief Initialize a `lbm_comm_t`:
/// - neighboors;
/// - size of the local mesh;
/// - relative position.
///
/// @param mesh_comm Mesh communicator to initialize.
/// @param rank Rank asking the initialization.
/// @param comm_size Size of the communicator.
/// @param width Width of the mesh.
/// @param height Height of the mesh.
void lbm_comm_init(lbm_comm_t* mesh_comm, int rank, int comm_size, uint32_t width, uint32_t height);

/// @brief Frees the memory of a `lbm_comm_t`.
/// @param mesh_comm Mesh communicator to free.
void lbm_comm_release(lbm_comm_t* mesh);

/// @brief Displays the configuration of the `lbm_comm_t` for a given rank.
/// @param mesh_comm Configuration to print.
void lbm_comm_print(const lbm_comm_t* mesh_comm);

/// @brief Blocking halo exchange of ghost cells.
void lbm_comm_halo_exchange(lbm_comm_t* mesh, Mesh* mesh_to_process);

/// @brief Post non-blocking Isend/Irecv for halo exchange. Call _wait() to complete.
void lbm_comm_halo_exchange_start(lbm_comm_t* mc, Mesh* m);

/// @brief Wait for non-blocking halo exchange to complete and unpack receive buffers.
void lbm_comm_halo_exchange_wait(lbm_comm_t* mc, Mesh* m);

/// @brief One-time halo exchange of cell types so that ghost cells carry the
///        correct boundary-condition type from the neighboring rank.
void lbm_comm_exchange_cell_types(lbm_comm_t* mc, lbm_mesh_type_t* mesh_type);

/// @brief Mesh rendering by doing reduction on rank 0 (master).
/// @param mesh_comm Communication mesh to use.
/// @param temp Temporary mesh to store the segments.
void save_frame_all_domain(FILE* fp, Mesh* source_mesh, Mesh* temp);
