#ifndef NAVIER_STOKES_CPP_ADVECTION_HPP
#define NAVIER_STOKES_CPP_ADVECTION_HPP

#include "utils.hpp"
#include "params.hpp"
#include "boundary.hpp"
#include "mesh.hpp"

#include <vector>
#include <stdexcept>
#include <numeric> 

namespace navier_stokes {


void compute_hllc_flux(Edge& edge, const SimulationParams& params, double& edge_flux_rho, double& edge_flux_rho_u, double& edge_flux_rho_v, double& edge_flux_E);


void computeAdvectionTerm(Mesh& mesh, const SimulationParams& params);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_ADVECTION_HPP

