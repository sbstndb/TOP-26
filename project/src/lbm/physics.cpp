#include <lbm/physics.hpp>

#include <cassert>
#include <cstdlib>

#include <omp.h>

#include <lbm/communications.hpp>
#include <lbm/config.hpp>
#include <lbm/structures.hpp>

// Tile size for loop blocking along the x (column) dimension.
// Override at compile time with -DTILE_X=N
// Set to 0 to disable tiling.
#ifndef TILE_X
#define TILE_X 32
#endif

#if DIRECTIONS == 9 && DIMENSIONS == 2
/// Definition of the 9 base vectors used to discretize the directions on each mesh.
const Vector direction_matrix[DIRECTIONS] = {
  // clang-format off
  {+0.0, +0.0},
  {+1.0, +0.0}, {+0.0, +1.0}, {-1.0, +0.0}, {+0.0, -1.0},
  {+1.0, +1.0}, {-1.0, +1.0}, {-1.0, -1.0}, {+1.0, -1.0},
  // clang-format on
};
#else
#error Need to define adapted direction matrix.
#endif

#if DIRECTIONS == 9
/// Weigths used to compensate the differences in lenght of the 9 directional vectors.
const double equil_weight[DIRECTIONS] = {
  // clang-format off
  4.0 / 9.0,
  1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
  // clang-format on
};

/// Opposite directions for bounce back implementation
const int opposite_of[DIRECTIONS] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
#else
#error Need to define adapted equilibrium distribution function
#endif

double get_vect_norm_2(Vector const a, Vector const b) {
  double res = 0.0;
  for (size_t k = 0; k < DIMENSIONS; k++) {
    res += a[k] * b[k];
  }
  return res;
}

double get_cell_density(const lbm_mesh_cell_t cell) {
  assert(cell != NULL);
  double res = 0.0;
  for (size_t k = 0; k < DIRECTIONS; k++) {
    res += cell[k];
  }
  return res;
}

void get_cell_velocity(Vector v, const lbm_mesh_cell_t cell, double cell_density) {
  assert(v != NULL);
  assert(cell != NULL);

  // Loop on all dimensions
  for (size_t d = 0; d < DIMENSIONS; d++) {
    v[d] = 0.0;

    // Sum all directions
    for (size_t k = 0; k < DIRECTIONS; k++) {
      v[d] += cell[k] * direction_matrix[k][d];
    }

    // Normalize
    v[d] /= cell_density;
  }
}

double compute_equilibrium_profile(Vector velocity, double density, int direction) {
  const double v2 = get_vect_norm_2(velocity, velocity);

  // Compute `e_i * v_i / c`
  const double p  = get_vect_norm_2(direction_matrix[direction], velocity);
  const double p2 = p * p;

  // Terms without density and direction weight
  double f_eq = 1.0 + (3.0 * p) + ((9.0 / 2.0) * p2) - ((3.0 / 2.0) * v2);

  // Multiply everything by the density and direction weight
  f_eq *= equil_weight[direction] * density;

  return f_eq;
}

void compute_cell_collision(lbm_mesh_cell_t cell_out, const lbm_mesh_cell_t cell_in) {
  // Compute macroscopic density
  double density = 0.0;
  for (int k = 0; k < DIRECTIONS; k++) {
    density += cell_in[k];
  }

  // Compute macroscopic velocity using D2Q9-specific formula (no direction_matrix lookup)
  const double inv_density = 1.0 / density;
  double vx = (cell_in[1] - cell_in[3] + cell_in[5] - cell_in[6] - cell_in[7] + cell_in[8]) * inv_density;
  double vy = (cell_in[2] - cell_in[4] + cell_in[5] + cell_in[6] - cell_in[7] - cell_in[8]) * inv_density;

  // Precompute v² (constant across all directions)
  const double v2 = vx * vx + vy * vy;
  const double relax = RELAX_PARAMETER;

  // Precompute dot products for all D2Q9 directions (no memory lookups)
  const double p[9] = {
    0.0,          // k=0: (0,0)
    vx,           // k=1: (1,0)
    vy,           // k=2: (0,1)
    -vx,          // k=3: (-1,0)
    -vy,          // k=4: (0,-1)
    vx + vy,      // k=5: (1,1)
    -vx + vy,     // k=6: (-1,1)
    -vx - vy,     // k=7: (-1,-1)
    vx - vy,      // k=8: (1,-1)
  };

  // Loop on microscopic directions with inlined equilibrium
  for (int k = 0; k < DIRECTIONS; k++) {
    const double f_eq = equil_weight[k] * density * (1.0 + 3.0 * p[k] + 4.5 * p[k] * p[k] - 1.5 * v2);
    cell_out[k] = cell_in[k] - relax * (cell_in[k] - f_eq);
  }
}

void compute_bounce_back(lbm_mesh_cell_t cell) {
  double tmp[DIRECTIONS];
  for (size_t k = 0; k < DIRECTIONS; k++) {
    tmp[k] = cell[opposite_of[k]];
  }
  for (size_t k = 0; k < DIRECTIONS; k++) {
    cell[k] = tmp[k];
  }
}

double helper_compute_poiseuille(const size_t i, const size_t size) {
  const double y = (double)(i - 1);
  const double L = (double)(size - 1);
  return 4.0 * INFLOW_MAX_VELOCITY / (L * L) * (L * y - y * y);
}

void compute_inflow_zou_he_poiseuille_distr(const Mesh* mesh, lbm_mesh_cell_t cell, size_t id_y) {
#if DIRECTIONS != 9
#error Implemented only for 9 directions
#endif

  // Set macroscopic fluid info
  // Poiseuille distribution on X and null on Y
  // We just want the norm, so `v = v_x`
  const double v = helper_compute_poiseuille(id_y, mesh->height);

  // Compute rho from U and inner flow on surface
  const double rho = (cell[0] + cell[2] + cell[4] + 2 * (cell[3] + cell[6] + cell[7])) / (1.0 - v);

  // Now compute unknown microscopic values
  cell[1] = cell[3]; // + (2.0/3.0) * density * v_y <--- no velocity on Y so v_y = 0
  cell[5] = cell[7] - (1.0 / 2.0) * (cell[2] - cell[4])
            + (1.0 / 6.0) * (rho * v); // + (1.0/2.0) * rho * v_y    <--- no velocity on Y so v_y = 0
  cell[8] = cell[6] + (1.0 / 2.0) * (cell[2] - cell[4])
            + (1.0 / 6.0) * (rho * v); //- (1.0/2.0) * rho * v_y    <--- no velocity on Y so v_y = 0

  // No need to copy already known one as the value will be "loss" in the wall at propagatation time
}

void compute_outflow_zou_he_const_density(lbm_mesh_cell_t cell) {
#if DIRECTIONS != 9
#error Implemented only for 9 directions
#endif

  double const rho = 1.0;
  // Compute macroscopic velocity depending on inner flow going onto the wall
  const double v = -1.0 + (1.0 / rho) * (cell[0] + cell[2] + cell[4] + 2 * (cell[1] + cell[5] + cell[8]));

  // Now can compute unknown microscopic values
  cell[3] = cell[1] - (2.0 / 3.0) * rho * v;
  cell[7] = cell[5]
            + (1.0 / 2.0) * (cell[2] - cell[4])
            // - (1.0/2.0) * (rho * v_y)    <--- no velocity on Y so v_y = 0
            - (1.0 / 6.0) * (rho * v);
  cell[6] = cell[8]
            + (1.0 / 2.0) * (cell[4] - cell[2])
            // + (1.0/2.0) * (rho * v_y)    <--- no velocity on Y so v_y = 0
            - (1.0 / 6.0) * (rho * v);
}

void special_cells(Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  // Loop on all inner cells — already outer i, inner j for cache locality
  for (size_t i = 1; i < mesh->width - 1; i++) {
    for (size_t j = 1; j < mesh->height - 1; j++) {
      switch (*(lbm_cell_type_t_get_cell(mesh_type, i, j))) {
      case CELL_FUILD:
        break;
      case CELL_BOUNCE_BACK:
        compute_bounce_back(Mesh_get_cell(mesh, i, j));
        break;
      case CELL_LEFT_IN:
        compute_inflow_zou_he_poiseuille_distr(mesh, Mesh_get_cell(mesh, i, j), j + mesh_comm->y);
        break;
      case CELL_RIGHT_OUT:
        compute_outflow_zou_he_const_density(Mesh_get_cell(mesh, i, j));
        break;
      }
    }
  }
}

void collision(Mesh* mesh_out, const Mesh* mesh_in) {
  assert(mesh_in->width == mesh_out->width);
  assert(mesh_in->height == mesh_out->height);

  // Loop on all inner cells — outer i, inner j for cache locality
  for (size_t i = 1; i < mesh_in->width - 1; i++) {
    for (size_t j = 1; j < mesh_in->height - 1; j++) {
      compute_cell_collision(Mesh_get_cell(mesh_out, i, j), Mesh_get_cell(mesh_in, i, j));
    }
  }
}

void special_cells_and_collision(Mesh* mesh_out, Mesh* mesh_in, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  assert(mesh_in->width == mesh_out->width);
  assert(mesh_in->height == mesh_out->height);

  const size_t w = mesh_in->width;
  const size_t h = mesh_in->height;

#if TILE_X > 0
  // Tiled fused pass: process TILE_X columns at a time for better L2 cache reuse
  for (size_t i0 = 1; i0 < w - 1; i0 += TILE_X) {
    const size_t i_end = (i0 + TILE_X < w - 1) ? (i0 + TILE_X) : (w - 1);
    for (size_t i = i0; i < i_end; i++) {
#else
  // Untiled fused pass
  {
    for (size_t i = 1; i < w - 1; i++) {
#endif
      for (size_t j = 1; j < h - 1; j++) {
        lbm_mesh_cell_t cell_in = Mesh_get_cell(mesh_in, i, j);

        // Apply boundary conditions in-place on the input cell
        switch (*(lbm_cell_type_t_get_cell(mesh_type, i, j))) {
        case CELL_FUILD:
          break;
        case CELL_BOUNCE_BACK:
          compute_bounce_back(cell_in);
          break;
        case CELL_LEFT_IN:
          compute_inflow_zou_he_poiseuille_distr(mesh_in, cell_in, j + mesh_comm->y);
          break;
        case CELL_RIGHT_OUT:
          compute_outflow_zou_he_const_density(cell_in);
          break;
        }

        // Now compute collision from (updated) mesh_in into mesh_out
        compute_cell_collision(Mesh_get_cell(mesh_out, i, j), cell_in);
      }
    }
  }
}

void propagation(Mesh* mesh_out, const Mesh* mesh_in) {
  const int w = mesh_out->width;
  const int h = mesh_out->height;

  // Gather pattern: for each destination cell, read from source neighbors
  // Direction offsets: direction k propagates FROM (i - dx[k], j - dy[k])
  static const int dir_x[DIRECTIONS] = {0, +1, 0, -1, 0, +1, -1, -1, +1};
  static const int dir_y[DIRECTIONS] = {0, 0, +1, 0, -1, +1, +1, -1, -1};

#if TILE_X > 0
  // Interior cells — tiled along x for L2 cache reuse, gather from neighbors
  for (int i0 = 1; i0 < w - 1; i0 += TILE_X) {
    const int i_end = (i0 + (int)TILE_X < w - 1) ? (i0 + (int)TILE_X) : (w - 1);
    for (int i = i0; i < i_end; i++) {
#else
  // Interior cells — untiled, gather from neighbors
  {
    for (int i = 1; i < w - 1; i++) {
#endif
      for (int j = 1; j < h - 1; j++) {
        double* __restrict__ cell_out = Mesh_get_cell(mesh_out, i, j);
        for (int k = 0; k < DIRECTIONS; k++) {
          cell_out[k] = Mesh_get_cell(mesh_in, i - dir_x[k], j - dir_y[k])[k];
        }
      }
    }
  }

  // Border cells — with bounds check, gather pattern
  // Top row (j=0)
  for (int i = 0; i < w; i++) {
    double* cell_out = Mesh_get_cell(mesh_out, i, 0);
    for (int k = 0; k < DIRECTIONS; k++) {
      int si = i - dir_x[k];
      int sj = 0 - dir_y[k];
      if (si >= 0 && si < w && sj >= 0 && sj < h)
        cell_out[k] = Mesh_get_cell(mesh_in, si, sj)[k];
    }
  }
  // Bottom row (j=h-1)
  for (int i = 0; i < w; i++) {
    double* cell_out = Mesh_get_cell(mesh_out, i, h - 1);
    for (int k = 0; k < DIRECTIONS; k++) {
      int si = i - dir_x[k];
      int sj = (h - 1) - dir_y[k];
      if (si >= 0 && si < w && sj >= 0 && sj < h)
        cell_out[k] = Mesh_get_cell(mesh_in, si, sj)[k];
    }
  }
  // Left column (i=0), skip corners
  for (int j = 1; j < h - 1; j++) {
    double* cell_out = Mesh_get_cell(mesh_out, 0, j);
    for (int k = 0; k < DIRECTIONS; k++) {
      int si = 0 - dir_x[k];
      int sj = j - dir_y[k];
      if (si >= 0 && si < w && sj >= 0 && sj < h)
        cell_out[k] = Mesh_get_cell(mesh_in, si, sj)[k];
    }
  }
  // Right column (i=w-1), skip corners
  for (int j = 1; j < h - 1; j++) {
    double* cell_out = Mesh_get_cell(mesh_out, w - 1, j);
    for (int k = 0; k < DIRECTIONS; k++) {
      int si = (w - 1) - dir_x[k];
      int sj = j - dir_y[k];
      if (si >= 0 && si < w && sj >= 0 && sj < h)
        cell_out[k] = Mesh_get_cell(mesh_in, si, sj)[k];
    }
  }
}

void collide_and_stream(Mesh* mesh_out, Mesh* mesh_in, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  assert(mesh_in->width == mesh_out->width);
  assert(mesh_in->height == mesh_out->height);

  const int w = (int)mesh_in->width;
  const int h = (int)mesh_in->height;

  // Direction offsets for scatter: direction k scatters TO (i + dir_x[k], j + dir_y[k])
  static const int dir_x[DIRECTIONS] = {0, +1, 0, -1, 0, +1, -1, -1, +1};
  static const int dir_y[DIRECTIONS] = {0, 0, +1, 0, -1, +1, +1, -1, -1};

  const double relax = RELAX_PARAMETER;

  // --- Main fused pass: BC + collision + scatter for inner source cells ---
  // Source cells are in [1, w-2] x [1, h-2].
  // Scatter destinations range from [0, w-1] x [0, h-1], always in bounds.

#if TILE_X > 0
  for (int i0 = 1; i0 < w - 1; i0 += TILE_X) {
    const int i_end = (i0 + (int)TILE_X < w - 1) ? (i0 + (int)TILE_X) : (w - 1);
    for (int i = i0; i < i_end; i++) {
#else
  {
    for (int i = 1; i < w - 1; i++) {
#endif
      for (int j = 1; j < h - 1; j++) {
        lbm_mesh_cell_t cell_in = Mesh_get_cell(mesh_in, i, j);

        // Apply boundary conditions in-place on cell_in
        switch (*(lbm_cell_type_t_get_cell(mesh_type, i, j))) {
        case CELL_FUILD:
          break;
        case CELL_BOUNCE_BACK:
          compute_bounce_back(cell_in);
          break;
        case CELL_LEFT_IN:
          compute_inflow_zou_he_poiseuille_distr(mesh_in, cell_in, j + mesh_comm->y);
          break;
        case CELL_RIGHT_OUT:
          compute_outflow_zou_he_const_density(cell_in);
          break;
        }

        // Compute collision (inlined D2Q9-specific)
        double density = 0.0;
        for (int k = 0; k < DIRECTIONS; k++) {
          density += cell_in[k];
        }
        const double inv_density = 1.0 / density;
        double vx = (cell_in[1] - cell_in[3] + cell_in[5] - cell_in[6] - cell_in[7] + cell_in[8]) * inv_density;
        double vy = (cell_in[2] - cell_in[4] + cell_in[5] + cell_in[6] - cell_in[7] - cell_in[8]) * inv_density;
        const double v2 = vx * vx + vy * vy;

        // Precompute dot products for all D2Q9 directions
        const double p[9] = {
          0.0, vx, vy, -vx, -vy, vx + vy, -vx + vy, -vx - vy, vx - vy,
        };

        // Collision + scatter: compute post-collision value and write to destination
        for (int k = 0; k < DIRECTIONS; k++) {
          const double f_eq = equil_weight[k] * density * (1.0 + 3.0 * p[k] + 4.5 * p[k] * p[k] - 1.5 * v2);
          const double collided = cell_in[k] - relax * (cell_in[k] - f_eq);
          Mesh_get_cell(mesh_out, i + dir_x[k], j + dir_y[k])[k] = collided;
        }
      }
    }
  }

  // --- Fixup pass: for any destination cell (i,j) and direction k where the  ---
  // --- source (i-dx[k], j-dy[k]) is NOT an inner cell, gather from mesh_in. ---
  // The scatter above only handles inner SOURCE cells [1,w-2]x[1,h-2].
  // Destination cells that need contributions from ghost/border source cells
  // include: border rows/cols (j=0, j=h-1, i=0, i=w-1) AND inner cells
  // adjacent to borders (j=1, j=h-2, i=1, i=w-2) for certain directions.
  //
  // We handle this by iterating over all cells in the 2-wide border strip and
  // fixing up only the directions whose source is not an inner cell.

  // Helper lambda: for destination (di, dj), fix directions with non-inner sources
  auto fixup_cell = [&](int di, int dj) {
    double* cell_out = Mesh_get_cell(mesh_out, di, dj);
    for (int k = 0; k < DIRECTIONS; k++) {
      int si = di - dir_x[k];
      int sj = dj - dir_y[k];
      // If source is an inner cell, scatter already wrote the correct value
      if (si >= 1 && si <= w - 2 && sj >= 1 && sj <= h - 2)
        continue;
      // Otherwise gather raw from mesh_in (no collision applied to ghost cells)
      if (si >= 0 && si < w && sj >= 0 && sj < h)
        cell_out[k] = Mesh_get_cell(mesh_in, si, sj)[k];
    }
  };

  // Top 2 rows (j=0, j=1)
  for (int j = 0; j <= 1 && j < h; j++) {
    for (int i = 0; i < w; i++) {
      fixup_cell(i, j);
    }
  }
  // Bottom 2 rows (j=h-2, j=h-1)
  for (int j = (h - 2 > 1 ? h - 2 : 2); j < h; j++) {
    for (int i = 0; i < w; i++) {
      fixup_cell(i, j);
    }
  }
  // Left 2 columns (i=0, i=1), skip rows already handled
  for (int i = 0; i <= 1 && i < w; i++) {
    for (int j = 2; j < h - 2; j++) {
      fixup_cell(i, j);
    }
  }
  // Right 2 columns (i=w-2, i=w-1), skip rows already handled
  for (int i = (w - 2 > 1 ? w - 2 : 2); i < w; i++) {
    for (int j = 2; j < h - 2; j++) {
      fixup_cell(i, j);
    }
  }
}
