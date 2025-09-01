#include "boundary.hpp"
#include <cassert>

namespace navier_stokes {


void apply_inflow_outflow_bc(const SimulationParams& params, const char boundaryCondition,
    double& rho, double& rho_u, double& rho_v, double& E) {

    double rho0  = params.rho_ref;
    double p0    = params.p_ref;
    double M     = params.inflow_velocity;
    double a0    = std::sqrt(params.gamma*p0/rho0);
    double u0    = M*a0;
    double v0    = 0.0;
    double kinetic = 0.5*rho0*(u0*u0 + v0*v0);
    double E0      = params.p_ref/(params.gamma-1.0) + kinetic;

    if (boundaryCondition == 'i') { // inlet
        rho   = rho0;
        rho_u = rho0 * u0;
        rho_v = rho0 * v0;
        E     = E0;
    }
    else if (boundaryCondition == 'o') { // outlet // on change rien
        // rho = rho, rho_u = rho_u, rho_v = rho_v, E = E;
    }
    else if (boundaryCondition == 'f') { // freestream // far field 
        rho   = rho0;
        rho_u = rho0 * u0;
        rho_v = rho0 * v0;
        E     = E0;
    }
    else if (boundaryCondition == 'w') {
        // rho = rho, rho_u = -rho_u, rho_v = -rho_v, E = E;
        // rho = rho, rho_u = 0, rho_v = 0, E = E; // no-slip
        const double kinetic_old = 0.5 * (rho_u*rho_u + rho_v*rho_v) / rho;
        const double e_internal  = E - kinetic_old; // e_internal = E_total - E_cinetique
        rho_u = 0.0; // no-slip + no-penetration
        rho_v = 0.0; // no-slip + no-penetration
        E     = e_internal; 
    }
    else 
        rho = rho, rho_u = rho_u, rho_v = rho_v, E = E;
}


void apply_InterfaceStates_boundary_conditions(const SimulationParams& params, const char boundaryCondition, double rho_L, double rho_u_L, double rho_v_L, double E_L, double& rho_R, double& rho_u_R, double& rho_v_R, double& E_R, double& nx, double& ny) {
    rho_R    = rho_L;  
    rho_u_R  = rho_u_L;   
    rho_v_R  = rho_v_L;   
    E_R      = E_L;

    if (boundaryCondition == 'i' || boundaryCondition == 'o' || boundaryCondition == 'f') {
        apply_inflow_outflow_bc(params, boundaryCondition, rho_R, rho_u_R, rho_v_R, E_R);
    }
    else if (boundaryCondition == 'w') {
        rho_u_R = -rho_u_L;
        rho_v_R = -rho_v_L;

        // double uL = rho_u_L / rho_L;
        // double vL = rho_v_L / rho_L;

        // double unL =  uL*nx + vL*ny;
        // double utL = -uL*ny + vL*nx;

        // double unR = -unL;
        // double utR =  utL;

        // double uR = unR*nx - utR*ny;
        // double vR = unR*ny + utR*nx;

        // double pL = (params.gamma-1.0)*(E_L - 0.5*rho_L*(uL*uL + vL*vL));
        // double pR = pL;
        // rho_R    = rho_L;
        // rho_u_R  = rho_R * uR;
        // rho_v_R  = rho_R * vR;
        // E_R      = pR/(params.gamma-1.0) + 0.5*rho_R*(uR*uR + vR*vR);
    }
}


//---------------------------------------------------------------------------
void apply_boundary_conditions(const SimulationParams& params, const char boundaryCondition,
    double& rho, double& rho_u, double& rho_v, double& E) {
    if (params.boundary_conditions == "inflow_outflow_x") {
        return apply_inflow_outflow_bc(params, boundaryCondition, rho, rho_u, rho_v, E);
    }
    else {
        throw std::invalid_argument("Unknown boundary condition: " + params.boundary_conditions);
    }
}







//---------------------------------------------------------------------------
void apply_diffusion_bc(const SimulationParams& params, const char boundaryCondition,
    double& tau_xx, double& tau_yy, double& tau_xy_yx,
    double& q_x, double& q_y) {
    switch (boundaryCondition) {
        case 'i': // inlet : rien
            break;
        case 'o': // outlet : rien
            break;
        case 'f': // far-field 
            tau_xx    = 0.0;
            tau_yy    = 0.0;
            tau_xy_yx = 0.0;
            q_x       = 0.0;
            q_y       = 0.0;
            break;
        case 'w': // paroi adiabatique 
            tau_xx    = 0.0;
            tau_yy    = 0.0;
            tau_xy_yx = 0.0;
            q_x       = 0.0;
            q_y       = 0.0;
            break;

        default:
            break;
    }
}









} // namespace navier_stokes




