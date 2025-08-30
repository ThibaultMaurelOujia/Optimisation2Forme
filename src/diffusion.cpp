#include "diffusion.hpp"

namespace navier_stokes {


void computeScalarDivergenceDiffusion(Mesh& mesh, const SimulationParams& params, 
    std::vector<double>& tau_xx, std::vector<double>& tau_yy, std::vector<double>& tau_xy_yx, std::vector<double>& q_x, std::vector<double>& q_y){

    static std::vector<double> vertices_tau_xx, vertices_tau_yy, vertices_tau_xy_yx, vertices_q_x, vertices_q_y;
    static std::vector<double> edges_tau_xx, edges_tau_yy, edges_tau_xy_yx, edges_q_x, edges_q_y;
    bool first = vertices_tau_xx.empty();
    if (first) {
        size_t N_vertices = mesh.vertices.size();
        vertices_tau_xx.resize(N_vertices);
        vertices_tau_yy.resize(N_vertices);
        vertices_tau_xy_yx.resize(N_vertices);
        vertices_q_x.resize(N_vertices);
        vertices_q_y.resize(N_vertices);
        size_t N_edges = mesh.edges.size();
        edges_tau_xx.resize(N_edges);
        edges_tau_yy.resize(N_edges);
        edges_tau_xy_yx.resize(N_edges);
        edges_q_x.resize(N_edges);
        edges_q_y.resize(N_edges);
    }

    auto t_v = time_tic();
    #pragma omp parallel for 
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        Vertex& vertex = mesh.vertices[i];
        if (vertex.cellIDs.size() == 3) {
            auto it = vertex.cellIDs.begin();
            std::size_t id0 = *it; ++it;
            std::size_t id1 = *it; ++it;
            std::size_t id2 = *it;

            Cell& cell0 = mesh.cells[id0];
            Cell& cell1 = mesh.cells[id1];
            Cell& cell2 = mesh.cells[id2];

            double w0_num = (cell1.centre.x - vertex.p.x) * (cell2.centre.y - vertex.p.y) - (cell2.centre.x - vertex.p.x) * (cell1.centre.y - vertex.p.y);
            double w0_den = (cell1.centre.x - cell0.centre.x) * (cell2.centre.y - cell0.centre.y) - (cell2.centre.x - cell0.centre.x) * (cell1.centre.y - cell0.centre.y);
            double w0 = (w0_num / w0_den);
            double w1_num = (cell2.centre.x - vertex.p.x) * (cell0.centre.y - vertex.p.y) - (cell0.centre.x - vertex.p.x) * (cell2.centre.y - vertex.p.y);
            double w1_den = (cell1.centre.x - cell0.centre.x) * (cell2.centre.y - cell0.centre.y) - (cell2.centre.x - cell0.centre.x) * (cell1.centre.y - cell0.centre.y);
            double w1 = (w1_num / w1_den);
            double w2 = 1 - w0 - w1;

            vertices_tau_xx[i]    = w0 * tau_xx[id0]    + w1 * tau_xx[id1]    + w2 * tau_xx[id2];
            vertices_tau_yy[i]    = w0 * tau_yy[id0]    + w1 * tau_yy[id1]    + w2 * tau_yy[id2];
            vertices_tau_xy_yx[i] = w0 * tau_xy_yx[id0] + w1 * tau_xy_yx[id1] + w2 * tau_xy_yx[id2];
            vertices_q_x[i]       = w0 * q_x[id0]       + w1 * q_x[id1]       + w2 * q_x[id2];
            vertices_q_y[i]       = w0 * q_y[id0]       + w1 * q_y[id1]       + w2 * q_y[id2];
        } 
        else if (vertex.cellIDs.size() == 4) {

            const auto& ord = vertex.cellIDsOrdered;
            std::array<std::size_t,4> cid = { ord[0], ord[1], ord[2], ord[3] };
            
            std::array<vector_3D,4>  C; 
            std::array<double,4>     vertices_tau_xx_;
            std::array<double,4>     vertices_tau_yy_;
            std::array<double,4>     vertices_tau_xy_yx_;
            std::array<double,4>     vertices_q_x_;
            std::array<double,4>     vertices_q_y_;

            for (int k = 0; k < 4; ++k) {
                const Cell& c = mesh.cells[cid[k]];
                C[k]                   = c.centre;
                vertices_tau_xx_[k]    = tau_xx[cid[k]]; 
                vertices_tau_yy_[k]    = tau_yy[cid[k]]; 
                vertices_tau_xy_yx_[k] = tau_xy_yx[cid[k]];  
                vertices_q_x_[k]       = q_x[cid[k]];  
                vertices_q_y_[k]       = q_y[cid[k]];  
            }

            std::array<vector_3D,4> U;
            std::array<double,4>    d; 

            for (int k = 0; k < 4; ++k) {
                double dx = C[k].x - vertex.p.x;
                double dy = C[k].y - vertex.p.y;
                d[k] = std::hypot(dx,dy);
                U[k] = { dx/d[k], dy/d[k], 0.0 };
            }

            std::array<double,4> w{};

            for (int k = 0; k < 4; ++k) {
                int km1 = (k+3) & 3; 
                int kp1 = (k+1) & 3; 

                auto dot  = [](const vector_3D& a,const vector_3D& b){ return a.x*b.x + a.y*b.y; };
                double cos_prev = std::clamp( dot(U[km1], U[k]), -1.0, 1.0 );
                double cos_next = std::clamp( dot(U[k],   U[kp1]), -1.0, 1.0 );

                double theta_prev = std::acos(cos_prev);
                double theta_next = std::acos(cos_next);

                w[k] = ( std::tan(0.5*theta_prev) + std::tan(0.5*theta_next) ) / d[k];
            }

            double wsum = std::accumulate(w.begin(), w.end(), 0.0);
            for (double& wk : w) wk /= wsum;

            vertices_tau_xx[i]    = w[0]*vertices_tau_xx_[0]    + w[1]*vertices_tau_xx_[1]    + w[2]*vertices_tau_xx_[2]    + w[3]*vertices_tau_xx_[3];
            vertices_tau_yy[i]    = w[0]*vertices_tau_yy_[0]    + w[1]*vertices_tau_yy_[1]    + w[2]*vertices_tau_yy_[2]    + w[3]*vertices_tau_yy_[3];
            vertices_tau_xy_yx[i] = w[0]*vertices_tau_xy_yx_[0] + w[1]*vertices_tau_xy_yx_[1] + w[2]*vertices_tau_xy_yx_[2] + w[3]*vertices_tau_xy_yx_[3]; 
            vertices_q_x[i]       = w[0]*vertices_q_x_[0]       + w[1]*vertices_q_x_[1]       + w[2]*vertices_q_x_[2]       + w[3]*vertices_q_x_[3];
            vertices_q_y[i]       = w[0]*vertices_q_y_[0]       + w[1]*vertices_q_y_[1]       + w[2]*vertices_q_y_[2]       + w[3]*vertices_q_y_[3];
        } 
        else if (vertex.cellIDs.size() == 2) {
            auto it = vertex.cellIDs.begin();
            std::size_t id0 = *it; ++it;
            std::size_t id1 = *it;

            const Cell& c0 = mesh.cells[id0];
            const Cell& c1 = mesh.cells[id1];

            // moyenne de base
            vertices_tau_xx[i]        = 0.5 * (tau_xx[id0]    + tau_xx[id1]);
            vertices_tau_yy[i]        = 0.5 * (tau_yy[id0]    + tau_yy[id1]);
            vertices_tau_xy_yx[i]     = 0.5 * (tau_xy_yx[id0] + tau_xy_yx[id1]);
            vertices_q_x[i]           = 0.5 * (q_x[id0]       + q_x[id1]);
            vertices_q_y[i]           = 0.5 * (q_y[id0]       + q_y[id1]);
        }
        else if (vertex.cellIDs.size() == 1) { 
            std::size_t id0 = *vertex.cellIDs.begin();
            const Cell& c0 = mesh.cells[id0];

            vertices_tau_xx[i]    = tau_xx[id0];
            vertices_tau_yy[i]    = tau_yy[id0];
            vertices_tau_xy_yx[i] = tau_xy_yx[id0]; 
            vertices_q_x[i]       = q_x[id0]; 
            vertices_q_y[i]       = q_y[id0]; 
        }
        if (vertex.boundaryCondition == 'i' || vertex.boundaryCondition == 'o' || vertex.boundaryCondition == 'f' || vertex.boundaryCondition == 'w') 
            apply_diffusion_bc(params, vertex.boundaryCondition, vertices_tau_xx[i], vertices_tau_yy[i], vertices_tau_xy_yx[i], vertices_q_x[i], vertices_q_y[i]);
            // apply_boundary_conditions(params, vertex.boundaryCondition, vertices_rho[i], vertices_rho_u[i], vertices_rho_v[i], vertices_E[i]); 
    }
    time_toc(t_v, "Interpolation vertex", params.verbosity); 


    auto t_e = time_tic();
    #pragma omp parallel for 
    for (std::size_t i = 0; i < mesh.edges.size(); ++i) {
        Edge& edge = mesh.edges[i];

        double edge_tau_xx    = 0.5 * (vertices_tau_xx[edge.leftVertexID]    + vertices_tau_xx[edge.rightVertexID]);
        double edge_tau_yy    = 0.5 * (vertices_tau_yy[edge.leftVertexID]    + vertices_tau_yy[edge.rightVertexID]);
        double edge_tau_xy_yx = 0.5 * (vertices_tau_xy_yx[edge.leftVertexID] + vertices_tau_xy_yx[edge.rightVertexID]); 
        double edge_q_x       = 0.5 * (vertices_q_x[edge.leftVertexID]       + vertices_q_x[edge.rightVertexID]); 
        double edge_q_y       = 0.5 * (vertices_q_y[edge.leftVertexID]       + vertices_q_y[edge.rightVertexID]); 
        
        if (edge.boundaryCondition == 'i' || edge.boundaryCondition == 'o' || edge.boundaryCondition == 'f' || edge.boundaryCondition == 'w') {
            apply_diffusion_bc(params, edge.boundaryCondition, edge_tau_xx, edge_tau_yy, edge_tau_xy_yx, edge_q_x, edge_q_y);
            // apply_boundary_conditions(params, edge.boundaryCondition, rho, rho_u, rho_v, E);  
            if (edge.boundaryCondition == 'w') {
                edge.tau_xx = edge_tau_xx; 
                edge.tau_xy_yx = edge_tau_xy_yx; 
                edge.tau_yy = edge_tau_yy; 
            }
        }
        
        edges_tau_xx[i]    = edge_tau_xx;
        edges_tau_yy[i]    = edge_tau_yy;
        edges_tau_xy_yx[i] = edge_tau_xy_yx; 
        edges_q_x[i]       = edge_q_x;
        edges_q_y[i]       = edge_q_y;
    }
    time_toc(t_e, "Interpolation edge", params.verbosity); 


    auto t_c = time_tic();
    /// Green-Gauss
    #pragma omp parallel for 
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        Cell& cell = mesh.cells[i];
        double cv = cell.cellVolume;
        double u  = cell.rho_u / cell.rho;
        double v  = cell.rho_v / cell.rho;

        for (std::size_t edgeID : cell.edgeIDs) {
            Edge& edge = mesh.edges[edgeID];

            double el  = edge.edgeLength;
            double enx = edge.edgeNormal.x;
            double eny = edge.edgeNormal.y;

            if (edge.leftCellID == i) {enx = -enx; eny = -eny;}

            double edge_tau_xx    = edges_tau_xx[edgeID];
            double edge_tau_yy    = edges_tau_yy[edgeID];
            double edge_tau_xy_yx = edges_tau_xy_yx[edgeID];
            double edge_q_x       = edges_q_x[edgeID];
            double edge_q_y       = edges_q_y[edgeID];
            cell.R_rho_u += edge_tau_xx * el * enx / cv + edge_tau_xy_yx * el * eny / cv;
            cell.R_rho_v += edge_tau_xy_yx * el * enx / cv + edge_tau_yy * el * eny / cv;
            cell.R_E     += (u * edge_tau_xx + v * edge_tau_xy_yx - edge_q_x) * el * enx / cv + (u * edge_tau_xy_yx + v * edge_tau_yy - edge_q_y) * el * eny / cv; 
        }
    }
    time_toc(t_c, "Gradient cell", params.verbosity); 
};


void computeDiffusionTerm(Mesh& mesh, const SimulationParams& params) {

    const double mu = params.viscosity;
    const double gamma = params.gamma;
    const double kappa = params.viscosity * params.cp / params.Pr;   

    static std::vector<double> tau_xx, tau_yy, tau_xy_yx, q_x, q_y;
    bool first = tau_xx.empty();
    if (first) {
        size_t N = mesh.cells.size();
        tau_xx.resize(N);
        tau_yy.resize(N);
        tau_xy_yx.resize(N);
        q_x.resize(N);
        q_y.resize(N);
    }

    #pragma omp parallel for 
    for(std::size_t cid=0; cid<mesh.cells.size(); ++cid){
        Cell& cell = mesh.cells[cid];
        
        double u   = cell.rho_u / cell.rho;
        double v   = cell.rho_v / cell.rho;
        double u_x = (cell.rho_u_x - u * cell.rho_x) / cell.rho;
        double v_y = (cell.rho_v_y - v * cell.rho_y) / cell.rho;
        double u_y = (cell.rho_u_y - u * cell.rho_y) / cell.rho;
        double v_x = (cell.rho_v_x - v * cell.rho_x) / cell.rho;

        tau_xx[cid] = 2 * mu * u_x - 2.0/3.0 * mu * (u_x + v_y);
        tau_yy[cid] = 2 * mu * v_y - 2.0/3.0 * mu * (u_x + v_y);
        tau_xy_yx[cid] = mu * (u_y + v_x);
        double T_x = (gamma - 1.0) * ((cell.E_x * cell.rho - cell.E * cell.rho_x) / (cell.rho * cell.rho) - (u * u_x + v * v_x));
        double T_y = (gamma - 1.0) * ((cell.E_y * cell.rho - cell.E * cell.rho_y) / (cell.rho * cell.rho) - (u * u_y + v * v_y));
        q_x[cid] = -kappa * T_x; 
        q_y[cid] = -kappa * T_y;
    }
    computeScalarDivergenceDiffusion(mesh, params, tau_xx, tau_yy, tau_xy_yx, q_x, q_y);
}



} // namespace navier_stokes