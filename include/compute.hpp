#ifndef NAVIER_STOKES_CPP_COMPUTE_HPP
#define NAVIER_STOKES_CPP_COMPUTE_HPP

#include "mesh.hpp"
#include "boundary.hpp"

#include <numeric> 

namespace navier_stokes {


void computeCellScalar(Mesh& mesh, const std::string& field, const SimulationParams& params);

void computeScalarVelocityX(Mesh& mesh);

void computeScalarVelocityMagnitude(Mesh& mesh);

void computeLiftDragCoefficients(const Mesh& mesh, const SimulationParams& params, double& C_L, double& C_D);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_COMPUTE_HPP