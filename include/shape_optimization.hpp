#ifndef NAVIER_STOKES_CPP_OPTIMIZATION_HPP
#define NAVIER_STOKES_CPP_OPTIMIZATION_HPP

#include "mesh.hpp"
// #include "boundary.hpp"
#include "advection.hpp"
#include "diffusion.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <numeric> 


namespace navier_stokes {


using SpMat    = Eigen::SparseMatrix<double>;
using Vec      = Eigen::VectorXd;
using Solver   = Eigen::BiCGSTAB<SpMat>;


void applyShapePerturbationNACA12(Mesh& mesh, const SimulationParams& params, const std::string& param_name, const std::vector<std::string>& param_names_all, const std::vector<double>& param_values_all, double eps = 1e-6);

void computeResidual(Mesh& mesh, const SimulationParams& params, 
                    std::vector<double>& R_rho, std::vector<double>& R_rho_u, std::vector<double>& R_rho_v, std::vector<double>& R_E);
    
void optimizeShape(Mesh& mesh, const SimulationParams& params, double eps = 1e-6);
                    
} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_OPTIMIZATION_HPP