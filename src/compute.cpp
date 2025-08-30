#include "compute.hpp"


namespace navier_stokes {


void computeCellScalar(Mesh& mesh, const std::string& field, const SimulationParams& params) {
    double scalar_min = +std::numeric_limits<double>::infinity();
    double scalar_max = -std::numeric_limits<double>::infinity();
    
    auto t_scalar = time_tic();
    #pragma omp parallel for reduction(min:scalar_min) reduction(max:scalar_max)
    for(std::size_t cid=0; cid<mesh.cells.size(); ++cid){
        Cell& cell = mesh.cells[cid];
        double u = 0.0, v = 0.0;
        
        if (cell.rho != 0.0) {
            u = cell.rho_u / cell.rho;
            v = cell.rho_v / cell.rho;
        }

        if      (field == "rho")      cell.scalar = cell.rho;
        else if (field == "rho_u")    cell.scalar = cell.rho_u;
        else if (field == "rho_v")    cell.scalar = cell.rho_v;
        else if (field == "E")        cell.scalar = cell.E;
        else if (field == "u")        cell.scalar = u;
        else if (field == "v")        cell.scalar = v;

        else if (field == "u_x") {
            cell.scalar = (cell.rho_u_x - u * cell.rho_x) / (cell.rho != 0.0 ? cell.rho : 1.0);
        }
        else if (field == "u_y") {
            cell.scalar = (cell.rho_u_y - u * cell.rho_y) / (cell.rho != 0.0 ? cell.rho : 1.0);
        }
        else if (field == "v_x") {
            cell.scalar = (cell.rho_v_x - v * cell.rho_x) / (cell.rho != 0.0 ? cell.rho : 1.0);
        }
        else if (field == "v_y") {
            cell.scalar = (cell.rho_v_y - v * cell.rho_y) / (cell.rho != 0.0 ? cell.rho : 1.0);
        }
        else if (field == "div") {
            double ux = (cell.rho_u_x - u * cell.rho_x) / (cell.rho != 0.0 ? cell.rho : 1.0);
            double vy = (cell.rho_v_y - v * cell.rho_y) / (cell.rho != 0.0 ? cell.rho : 1.0);
            cell.scalar = ux + vy;
        }
        else if (field == "p"  || field == "pressure") {
            double kinetic = 0.5 * (cell.rho_u*cell.rho_u + cell.rho_v*cell.rho_v) / cell.rho;
            double p_internal = (params.gamma - 1.0) * (cell.E - kinetic);
            cell.scalar = p_internal;
        }
        else {
            throw std::invalid_argument("Champ inconnu pour updateCellScalar: " + field);
        }
        
        if (scalar_min > cell.scalar) scalar_min = cell.scalar;
        if (scalar_max < cell.scalar) scalar_max = cell.scalar;
    }
    mesh.scalar_min = scalar_min;
    mesh.scalar_max = scalar_max;
    time_toc(t_scalar, "computeCellScalar", params.verbosity); 
}


void computeScalarVelocityX(Mesh& mesh)
{
    for (Cell& cell : mesh.cells) {
        if (cell.rho != 0.0) {
            cell.scalar = cell.rho_u / cell.rho;
        } else {
            cell.scalar = 0.0; // valeur par défaut
        }
    }
}


void computeScalarVelocityMagnitude(Mesh& mesh)
{
    for (Cell& cell : mesh.cells) {
        if (cell.rho != 0.0) {
            double ux = cell.rho_u / cell.rho;
            double uy = cell.rho_v / cell.rho;
            cell.scalar = std::sqrt(ux*ux + uy*uy);
        } else {
            cell.scalar = 0.0;
        }
    }
}


inline double sound_speed_ideal(double gamma, double p, double rho){ return std::sqrt(gamma * p / rho); }

void computeLiftDragCoefficients(const Mesh& mesh, const SimulationParams& params, double& C_L, double& C_D) {
    double gamma = params.gamma;
    double rho_inf = params.rho_ref;
    double p_inf   = params.p_ref; 
    const double a_inf   = sound_speed_ideal(gamma,p_inf,rho_inf);
    double M_inf  = params.inflow_velocity;   
    double U_inf = M_inf * a_inf;

    double q_inf = 0.5 * (rho_inf * U_inf*U_inf);
    double corde = 1; // !!! Attention : à adapter

    double L = 0.0, D = 0.0;
    
    #pragma omp parallel for reduction(+:L,D)
    for (size_t i = 0; i < mesh.edges.size(); ++i) {
        const Edge& edge = mesh.edges[i];
        
        if (edge.boundaryCondition != 'w') continue;
        
        // if (edge.centre.x > 1.34) continue; // !!! test !!!

        const auto& n = edge.edgeNormal;
        double A = edge.edgeLength;

        // pression 
        double p = (params.gamma - 1.0) * (edge.E_L - 0.5*(edge.rho_u_L*edge.rho_u_L + edge.rho_v_L*edge.rho_v_L) / edge.rho_L);

        // tau
        double tau_xx = edge.tau_xx,
               tau_xy_yx = edge.tau_xy_yx,
               tau_yy = edge.tau_yy;

        L += (-p * n.y + tau_xy_yx * n.x + tau_yy * n.y) * A;
        D += (-p * n.x + tau_xx * n.x + tau_xy_yx * n.y) * A;
        // L += (-p * n.y) * A;
        // D += (-p * n.x) * A;
    }

    C_L = L / (q_inf * corde);
    C_D = D / (q_inf * corde);
}








// // OLD
// //---------------------------------------------------------------------------

// void compute_vorticity(
//     const Field& rho, const Field& rho_u, const Field& rho_v, 
//     Field& w, 
//     int Nx, int Ny, int N_ghost,
//     double dx, double dy) { 
//     if ((int)rho.size()   != Nx+2*N_ghost ||
//         (int)rho_u.size() != Nx+2*N_ghost ||
//         (int)rho_v.size() != Nx+2*N_ghost ||
//         (int)w.size()     != Nx+2*N_ghost ||
//         (int)rho.front().size()   != Ny+2*N_ghost ||
//         (int)rho_u.front().size() != Ny+2*N_ghost ||
//         (int)rho_v.front().size() != Ny+2*N_ghost ||
//         (int)w.front().size()     != Ny+2*N_ghost)
//     {
//         throw std::invalid_argument("Tailles de champs incorrectes dans compute_vorticity");
//     }
 
//     #pragma omp parallel for collapse(2)
//     for(int i = N_ghost; i < Nx+N_ghost; ++i) {
//         for(int j = N_ghost; j < Ny+N_ghost; ++j) { 
//             double u_c  = rho_u[i][j]   / rho[i][j];
//             double v_c  = rho_v[i][j]   / rho[i][j];
//             double v_ip = rho_v[i+1][j] / rho[i+1][j];
//             double v_im = rho_v[i-1][j] / rho[i-1][j];
//             double u_jp = rho_u[i][j+1] / rho[i][j+1];
//             double u_jm = rho_u[i][j-1] / rho[i][j-1];

//             double dvdx = (v_ip - v_im) / (2.0 * dx);
//             double dudy = (u_jp - u_jm) / (2.0 * dy);
//             w[i][j]    = dvdx - dudy;
//         }
//     }
 
//     for(int i = 0; i < N_ghost; ++i) {
//         for(int j = 0; j < (int)w[i].size(); ++j) {
//             w[i][j]               = 0.0;
//             w[Nx+N_ghost][j]      = 0.0;
//         }
//     }
//     for(int i = 0; i < (int)w.size(); ++i) {
//         for(int j = 0; j < N_ghost; ++j) {
//             w[i][j]               = 0.0;
//             w[i][Ny+N_ghost]      = 0.0;
//         }
//     }
// }

// void compute_schlieren(
//     const Field& rho,
//     Field& schlieren,
//     int Nx, int Ny, int N_ghost,
//     double dx, double dy) { 
//     if ((int)rho.size()     != Nx+2*N_ghost ||
//         (int)schlieren.size() != Nx+2*N_ghost ||
//         (int)rho.front().size()     != Ny+2*N_ghost ||
//         (int)schlieren.front().size() != Ny+2*N_ghost)
//     {
//         throw std::invalid_argument("Tailles de champs incorrectes pour schlieren");
//     }
 
//     #pragma omp parallel for collapse(2)
//     for(int i = N_ghost; i < Nx+N_ghost; ++i) {
//         for(int j = N_ghost; j < Ny+N_ghost; ++j) {
//             double drdx = (rho[i+1][j] - rho[i-1][j]) / (2.0 * dx);
//             double drdy = (rho[i][j+1] - rho[i][j-1]) / (2.0 * dy);
//             schlieren[i][j] = std::sqrt(drdx*drdx + drdy*drdy);
//         }
//     }
 
//     for(int i = 0; i < N_ghost; ++i) {
//         for(int j = 0; j < (int)schlieren[i].size(); ++j) {
//             schlieren[i][j]          = 0.0;
//             schlieren[Nx+N_ghost][j] = 0.0;
//         }
//     }
//     for(int i = 0; i < (int)schlieren.size(); ++i) {
//         for(int j = 0; j < N_ghost; ++j) {
//             schlieren[i][j]          = 0.0;
//             schlieren[i][Ny+N_ghost] = 0.0;
//         }
//     }
// }






} // namespace navier_stokes













