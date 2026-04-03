#include <lbm/structures.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

void Mesh_init(Mesh* mesh, uint32_t width, uint32_t height) {
  // Setup parameters
  mesh->width  = width;
  mesh->height = height;

  // Allocate memory for cells (64-byte aligned for cache line / AVX-512)
  size_t cells_size = width * height * DIRECTIONS * sizeof(double);
  cells_size = (cells_size + 63) & ~static_cast<size_t>(63); // round up to multiple of 64
  mesh->cells = static_cast<double*>(aligned_alloc(64, cells_size));
}

void Mesh_release(Mesh* mesh) {
  mesh->width  = 0;
  mesh->height = 0;
  free(mesh->cells);
}

void lbm_mesh_type_t_init(lbm_mesh_type_t* meshtype, uint32_t width, uint32_t height) {
  // Setup parameters
  meshtype->width  = width;
  meshtype->height = height;

  // Allocate memory for cells (64-byte aligned for cache line / AVX-512)
  size_t types_size = (width + 2) * height * sizeof(lbm_cell_type_t);
  types_size = (types_size + 63) & ~static_cast<size_t>(63); // round up to multiple of 64
  meshtype->types = static_cast<lbm_cell_type_t*>(aligned_alloc(64, types_size));
  if (meshtype->types == NULL) {
    perror("malloc");
    abort();
  }
}

void lbm_mesh_type_t_release(lbm_mesh_type_t* mesh) {
  mesh->width  = 0;
  mesh->height = 0;
  free(mesh->types);
}

void fatal(const char* message) {
  fprintf(stderr, "FATAL ERROR : %s\n", message);
  abort();
}
