#ifndef NAVIER_STOKES_CPP_DIFFUSION_HPP
#define NAVIER_STOKES_CPP_DIFFUSION_HPP

#include "utils.hpp"
#include "params.hpp"
#include "boundary.hpp"
#include "mesh.hpp"

#include <vector>
#include <stdexcept>
#include <numeric> 

namespace navier_stokes {


void computeDiffusionTerm(Mesh& mesh, const SimulationParams& params);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_DIFFUSION_HPP

