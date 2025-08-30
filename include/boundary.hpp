#ifndef NAVIER_STOKES_CPP_BOUNDARY_HPP
#define NAVIER_STOKES_CPP_BOUNDARY_HPP

#include "utils.hpp"
#include "params.hpp"
#include "boundary.hpp"

#include <vector>

namespace navier_stokes {


using ObstacleBorderCell = std::pair<int,int>;

struct ObstacleCutCell {
    int I, J; 
    double alpha;
    double nx, ny; 
};

struct ObstacleMask {
    std::vector<ObstacleBorderCell> obstacle;

    std::vector<ObstacleBorderCell> left;
    std::vector<ObstacleBorderCell> right;
    std::vector<ObstacleBorderCell> bottom;
    std::vector<ObstacleBorderCell> top;

    std::vector<std::vector<bool>> solid;

    std::vector<ObstacleCutCell> cuts;
};

ObstacleMask createMask(const SimulationParams& p);


void apply_boundary_conditions(const SimulationParams& params, const char boundaryCondition, double& rho, double& rho_u, double& rho_v, double& E);

void apply_diffusion_bc(const SimulationParams& params, const char boundaryCondition, double& tau_xx, double& tau_yy, double& tau_xy_yx, double& q_x, double& q_y);

void apply_InterfaceStates_boundary_conditions(const SimulationParams& params, const char boundaryCondition, 
    double  rho_L, double  rho_u_L, double  rho_v_L, double  E_L, 
    double& rho_R, double& rho_u_R, double& rho_v_R, double& E_R);

    
} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_BOUNDARY_HPP