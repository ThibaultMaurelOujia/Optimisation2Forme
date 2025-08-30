#ifndef NAVIER_STOKES_CPP_INIT_HPP
#define NAVIER_STOKES_CPP_INIT_HPP

#include "utils.hpp"
#include "mesh.hpp"
#include "params.hpp"

#include <vector>
#include <cmath>
#include <cassert>
#include <random>

namespace navier_stokes {


void initialize_flow_field(Mesh& mesh, const SimulationParams& params);


void computeDivergence_sin_cos(Mesh& mesh, const SimulationParams& params);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_INIT_HPP