// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Conlain Kelly, Nic Olsen, Dan Negrut
// =============================================================================

#pragma once

#include <climits>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <functional>

namespace chrono {
namespace gpu {

/// Used to compute position as a function of time.
typedef std::function<double3(float)> GranPositionFunction;

/// Position function representing no motion or offset as a funtion of time.
const GranPositionFunction GranPosFunction_default = [](float t) { return make_double3(0, 0, 0); };

/// Verbosity level of the system.
enum class CHGPU_VERBOSITY { QUIET = 0, INFO = 1, METRICS = 2 };

/// Verbosity level.
enum class CHGPU_MESH_VERBOSITY { QUIET = 0, INFO = 1 };

/// Output mode of system.
enum class CHGPU_OUTPUT_MODE { CSV, BINARY, HDF5, NONE };

/// How are we integrating through time.
enum class CHGPU_TIME_INTEGRATOR { FORWARD_EULER, CHUNG, CENTERED_DIFFERENCE, EXTENDED_TAYLOR };

/// Supported friction model.
enum class CHGPU_FRICTION_MODE { FRICTIONLESS, SINGLE_STEP, MULTI_STEP };

/// Rolling resistance models -- ELASTIC_PLASTIC not implemented yet.
enum class CHGPU_ROLLING_MODE { NO_RESISTANCE, SCHWARTZ, ELASTIC_PLASTIC };

enum CHGPU_OUTPUT_FLAGS { ABSV = 1 << 0, VEL_COMPONENTS = 1 << 1,
                          FIXITY = 1 << 2, ANG_VEL_COMPONENTS = 1 << 3,
                          FORCE_COMPONENTS = 1 << 4, TYPE = 1 << 5,
                          CLUSTER = 1 << 6, ADJACENCY = 1 << 7 };

/// Clustering algorithm switches
/// 0 is reserved for NONE
/// 0 on CLUSTER_GRAPH_METHOD or CLUSTER_SEARCH_METHOD switch means no clustering
/// 0 on CLUSTER_GROUND_METHOD means GROUND cluster not tagged

enum class CLUSTER_GRAPH_METHOD {NONE = 0, CONTACT = 1, PROXIMITY = 2}; /* TODO: Implement proximity graph construction */
/// CONTACT leverages sphere_contact_map to build the graph.
/// PROXIMITY determine contacts by checking if distance between sphere pairs < gbscan_radius; TODO UNTESTED

enum class CLUSTER_SEARCH_METHOD {NONE = 0, BFS = 1}; // TO DO: implement faster search than BFS
/// BFS -> Breadth-First search

enum class CLUSTER_GROUND_METHOD {NONE = 0, BIGGEST = 1, LOWEST = 2};
/// BIGGEST: cluster with the most number of spheres is GROUND
/// LOWEST: clusters with any particle below a z-plane are GROUND

/// Cluster index.
/// NOISE spheres are part of the INVALID cluster (not really a cluster)
/// Any cluster that contain a VOLUME sphere is the VOLUME, EXCEPT the GROUND
/// Otherwise cluster index increases from START when clusters found 
/// no sphere with NONE if no bugs.
enum class CLUSTER_INDEX {NONE = 0, GROUND = 1, INVALID = 2, VOLUME = 3, START = 4}; /* number of clusters go up to nSpheres */

/// Sphere type, CORE BORDER or NOISE for clustering, VOLUME for inside mesh
/// a sphere can be a CORE or BORDER of any cluster. 
/// no sphere with NONE if no bugs.
enum class SPHERE_TYPE {NONE = 0, CORE = 1, BORDER = 2, NOISE = 3, VOLUME = 4};

#define GET_OUTPUT_SETTING(setting) (this->output_flags & static_cast<unsigned int>(setting))

}  // namespace gpu
}  // namespace chrono

typedef longlong3 int64_t3;

constexpr size_t BD_WALL_ID_X_BOT = 0;
constexpr size_t BD_WALL_ID_X_TOP = 1;
constexpr size_t BD_WALL_ID_Y_BOT = 2;
constexpr size_t BD_WALL_ID_Y_TOP = 3;
constexpr size_t BD_WALL_ID_Z_BOT = 4;
constexpr size_t BD_WALL_ID_Z_TOP = 5;

constexpr size_t NUM_RESERVED_BC_IDS = 6;

/// At most 8 domains are touched by a sphere
#define MAX_SDs_TOUCHED_BY_SPHERE 8
/// At most 8 domains are touched by a sphere
#define MAX_SPHERES_TOUCHED_BY_SPHERE 12
/// The L-size of an SD should contain, on average, about these many spheres
#define AVERAGE_SPHERES_PER_SD_X_DIR 3.5
/// The D-size of an SD should contain, on average, about these many spheres
#define AVERAGE_SPHERES_PER_SD_Y_DIR 3.5
/// The H-size of an SD should contain, on average, about these many spheres
#define AVERAGE_SPHERES_PER_SD_Z_DIR 3.5
/// Anticipated max number of DEs in an SD; used for setting aside memory ahead of time
#define MAX_COUNT_OF_SPHERES_PER_SD 256
/// Value that indicates non-valid ID. The assumption is that an ID is always a positive integer
#define NULL_CHGPU_ID UINT_MAX
/// Value that controls the length unit. A sphere deforms this many simulation length units under its own weight.
#define PSI_L_DEFAULT 16
/// Fraction of sphere radius which gives an upper bound on the length unit.
#define PSI_R_DEFAULT 1.f
/// Value that controls the time unit -- safety factor on the deformation-based time unit
#define PSI_T_DEFAULT 32
/// Maximum number of triangles that an SD can touch
#define MAX_TRIANGLE_COUNT_PER_SD 512u
/// Number of threads in a block when that number is allowed to vary.
#define CUDA_THREADS_PER_BLOCK 128

// NOTE this may change in the future, but until then this is sufficient
constexpr int warp_size = 32;

/// Set up some error checking mechanism to ensure CUDA didn't complain about things.
///  This approach suggested <a
/// href="https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api">elsewhere</a>.
///  Some nice suggestions for how to use the mechanism are provided at the above link.
///
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// Add verbose checks easily
#define INFO_PRINTF(...)                                                               \
    if (verbosity == CHGPU_VERBOSITY::INFO || verbosity == CHGPU_VERBOSITY::METRICS) { \
        printf(__VA_ARGS__);                                                           \
    }

#define MESH_INFO_PRINTF(...)                           \
    if (mesh_verbosity == CHGPU_MESH_VERBOSITY::INFO) { \
        printf(__VA_ARGS__);                            \
    }

#define METRICS_PRINTF(...)                      \
    if (verbosity == CHGPU_VERBOSITY::METRICS) { \
        printf(__VA_ARGS__);                     \
    }

#define CONDITIONAL_PRINTF(do_print, ...) \
    if (do_print) {                       \
        printf(__VA_ARGS__);              \
    }
