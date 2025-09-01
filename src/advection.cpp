#include "advection.hpp"

namespace navier_stokes {


void computeScalarGradientAdvection(Mesh& mesh, const SimulationParams& params){


    static std::vector<double> vertices_rho, vertices_rho_u, vertices_rho_v, vertices_E;
    static std::vector<double> edges_rho, edges_rho_u, edges_rho_v, edges_E;
    bool first = vertices_rho.empty();
    if (first) {
        size_t N_vertices = mesh.vertices.size();
        vertices_rho.resize(N_vertices);
        vertices_rho_u.resize(N_vertices);
        vertices_rho_v.resize(N_vertices);
        vertices_E.resize(N_vertices);
        size_t N_edges = mesh.edges.size();
        edges_rho.resize(N_edges);
        edges_rho_u.resize(N_edges);
        edges_rho_v.resize(N_edges);
        edges_E.resize(N_edges);
    }

    auto t_v = time_tic();
    #pragma omp parallel for //schedule(static, 64) //collapse(2) if(!omp_in_parallel()) schedule(static, 64) // schedule(dynamic)
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

            vertices_rho[i]   = w0 * cell0.rho   + w1 * cell1.rho   + w2 * cell2.rho;
            vertices_rho_u[i] = w0 * cell0.rho_u + w1 * cell1.rho_u + w2 * cell2.rho_u;
            vertices_rho_v[i] = w0 * cell0.rho_v + w1 * cell1.rho_v + w2 * cell2.rho_v;
            vertices_E[i]     = w0 * cell0.E     + w1 * cell1.E     + w2 * cell2.E; 
        } 
        else if (vertex.cellIDs.size() == 4) {

            const auto& ord = vertex.cellIDsOrdered;
            std::array<std::size_t,4> cid = { ord[0], ord[1], ord[2], ord[3] };
            
            std::array<vector_3D,4>  C; 
            std::array<double,4>     rhoV; 
            std::array<double,4>     rho_uV;
            std::array<double,4>     rho_vV;
            std::array<double,4>     EV;

            for (int k = 0; k < 4; ++k) {
                const Cell& c = mesh.cells[cid[k]];
                C[k]      = c.centre;
                rhoV[k]   = c.rho;
                rho_uV[k] = c.rho_u;
                rho_vV[k] = c.rho_v;
                EV[k]     = c.E;
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

            vertices_rho[i]   = w[0]*rhoV[0]   + w[1]*rhoV[1]   + w[2]*rhoV[2]   + w[3]*rhoV[3];
            vertices_rho_u[i] = w[0]*rho_uV[0] + w[1]*rho_uV[1] + w[2]*rho_uV[2] + w[3]*rho_uV[3];
            vertices_rho_v[i] = w[0]*rho_vV[0] + w[1]*rho_vV[1] + w[2]*rho_vV[2] + w[3]*rho_vV[3];
            vertices_E[i]     = w[0]*EV[0]     + w[1]*EV[1]     + w[2]*EV[2]     + w[3]*EV[3]; 
        } 
        else if (vertex.cellIDs.size() == 2) {
            auto it = vertex.cellIDs.begin();
            std::size_t id0 = *it; ++it;
            std::size_t id1 = *it;

            const Cell& c0 = mesh.cells[id0];
            const Cell& c1 = mesh.cells[id1];

            // moyenne de base
            vertices_rho[i]   = 0.5 * (c0.rho   + c1.rho);
            vertices_rho_u[i] = 0.5 * (c0.rho_u + c1.rho_u);
            vertices_rho_v[i] = 0.5 * (c0.rho_v + c1.rho_v);
            vertices_E[i]     = 0.5 * (c0.E     + c1.E); 
        }
        else if (vertex.cellIDs.size() == 1) { 
            std::size_t id0 = *vertex.cellIDs.begin();
            const Cell& c0 = mesh.cells[id0];

            vertices_rho[i]   = c0.rho;
            vertices_rho_u[i] = c0.rho_u;
            vertices_rho_v[i] = c0.rho_v;
            vertices_E[i]     = c0.E; 
        }
        if (vertex.boundaryCondition == 'i' || vertex.boundaryCondition == 'o' || vertex.boundaryCondition == 'f' || vertex.boundaryCondition == 'w')
            apply_boundary_conditions(params, vertex.boundaryCondition, vertices_rho[i], vertices_rho_u[i], vertices_rho_v[i], vertices_E[i]); 
    }
    time_toc(t_v, "Interpolation vertex", params.verbosity); 


    auto t_e = time_tic();
    #pragma omp parallel for 
    for (std::size_t i = 0; i < mesh.edges.size(); ++i) {
        Edge& edge = mesh.edges[i];

        double rho   = 0.5 * (vertices_rho[edge.leftVertexID]   + vertices_rho[edge.rightVertexID]);
        double rho_u = 0.5 * (vertices_rho_u[edge.leftVertexID] + vertices_rho_u[edge.rightVertexID]);
        double rho_v = 0.5 * (vertices_rho_v[edge.leftVertexID] + vertices_rho_v[edge.rightVertexID]);
        double E     = 0.5 * (vertices_E[edge.leftVertexID]     + vertices_E[edge.rightVertexID]); 
        
        if (edge.boundaryCondition == 'i' || edge.boundaryCondition == 'o' || edge.boundaryCondition == 'f' || edge.boundaryCondition == 'w') 
            apply_boundary_conditions(params, edge.boundaryCondition, rho, rho_u, rho_v, E);
        
        edges_rho[i]   = rho;
        edges_rho_u[i] = rho_u;
        edges_rho_v[i] = rho_v;
        edges_E[i]     = E; 
    }
    time_toc(t_e, "Interpolation edge", params.verbosity); 


    auto t_c = time_tic();
    #pragma omp parallel for 
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        Cell& cell = mesh.cells[i];
        cell.rho_x   = cell.rho_y   = 0.0;
        cell.rho_u_x = cell.rho_u_y = 0.0;
        cell.rho_v_x = cell.rho_v_y = 0.0;
        cell.E_x     = cell.E_y     = 0.0;
    }
    /// Green-Gauss
    #pragma omp parallel for 
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        Cell& cell = mesh.cells[i];
        double cv = cell.cellVolume;
        for (std::size_t edgeID : cell.edgeIDs) {
            Edge& edge = mesh.edges[edgeID];

            double el  = edge.edgeLength;
            double enx = edge.edgeNormal.x;
            double eny = edge.edgeNormal.y;

            if (edge.leftCellID == i) {enx = -enx; eny = -eny;}

            cell.rho_x   += edges_rho[edgeID] * el * enx / cv; 
            cell.rho_u_x += edges_rho_u[edgeID] * el * enx / cv;  
            cell.rho_v_x += edges_rho_v[edgeID] * el * enx / cv;  
            cell.E_x     += edges_E[edgeID] * el * enx / cv;  
            cell.rho_y   += edges_rho[edgeID] * el * eny / cv; 
            cell.rho_u_y += edges_rho_u[edgeID] * el * eny / cv; 
            cell.rho_v_y += edges_rho_v[edgeID] * el * eny / cv;  
            cell.E_y     += edges_E[edgeID] * el * eny / cv;  
        }
    }
    time_toc(t_c, "Gradient cell", params.verbosity); 
    
};



static inline double computeBarthJespersenAlpha(double phi_i, double phi_pred, double phi_min, double phi_max) {
    double alpha = 1.0;
    if (std::abs(phi_pred - phi_i) < std::numeric_limits<double>::epsilon())
        return alpha;

    if (phi_pred > phi_i) {
        alpha = std::min(1.0, (phi_max - phi_i) / (phi_pred - phi_i));
    }
    else if (phi_pred < phi_i) {
        alpha = std::min(1.0, (phi_min - phi_i) / (phi_pred - phi_i));
    }
    return alpha;
}

static inline double computeVenkatAlpha(double phi_i, double phi_pred, double phi_min, double phi_max, double cellArea, double k_eps = 1e-2) {
    const double delta = phi_pred - phi_i;
    if (std::abs(delta) < std::numeric_limits<double>::epsilon())
        return 1.0;

    const double h   = std::sqrt(cellArea / M_PI);      // rayon équivalent
    const double eps = k_eps*k_eps*k_eps * h*h*h;       // (k h)^3

    auto psi = [eps](double a, double d) -> double
    {
        return (a * a + eps + 2.0 * a * d) /
               (a * a + eps + 2.0 * d * d);
    };

    double alpha = 1.0;
    if (delta > 0.0)
        alpha = psi(phi_max - phi_i, delta);
    else
        alpha = psi(phi_min - phi_i, delta);

    return std::clamp(alpha, 0.0, 1.0);
}

void computeInterfaceStates(Mesh& mesh, const SimulationParams& params) {
    // std::cout << "ATTENTION IL FAUT REMETTRE computeVenkatAlpha" << '\n' << std::flush;
    
    auto t_inter = time_tic();
    #pragma omp parallel for 
    for (std::size_t cid = 0; cid < mesh.cells.size(); ++cid) {
        Cell& cell = mesh.cells[cid];

        double rho_min   = cell.rho,   rho_max   = cell.rho;
        double rho_u_min = cell.rho_u, rho_u_max = cell.rho_u;
        double rho_v_min = cell.rho_v, rho_v_max = cell.rho_v;
        double E_min     = cell.E,     E_max     = cell.E;

        for (std::size_t neighbourID : cell.neighbourIDs) {   
            const Cell& nb = mesh.cells[neighbourID];

            if (nb.rho   < rho_min)   rho_min   = nb.rho;
            if (nb.rho   > rho_max)   rho_max   = nb.rho;

            if (nb.rho_u < rho_u_min) rho_u_min = nb.rho_u;
            if (nb.rho_u > rho_u_max) rho_u_max = nb.rho_u;

            if (nb.rho_v < rho_v_min) rho_v_min = nb.rho_v;
            if (nb.rho_v > rho_v_max) rho_v_max = nb.rho_v;

            if (nb.E     < E_min)     E_min     = nb.E;
            if (nb.E     > E_max)     E_max     = nb.E;
        }

        double cell_x = cell.centre.x, cell_y = cell.centre.y;
        double cell_vol = cell.cellVolume;
        // double alpha = 1.0;
        double alpha_rho = 1.0, alpha_rho_u = 1.0, alpha_rho_v = 1.0, alpha_E = 1.0;
        double phi_i, phi_pred;
        for (std::size_t edgeID : cell.edgeIDs) {   
            Edge& edge = mesh.edges[edgeID];
            double edge_x = edge.centre.x, edge_y = edge.centre.y;
            double distance_x = edge_x - cell_x, distance_y = edge_y - cell_y;

            phi_i = cell.rho;
            phi_pred = phi_i + cell.rho_x * distance_x + cell.rho_y * distance_y;
            // alpha = std::min(alpha, computeBarthJespersenAlpha(phi_i, phi_pred, rho_min, rho_max));
            // alpha_rho = std::min(alpha_rho, computeBarthJespersenAlpha(phi_i, phi_pred, rho_min, rho_max));
            alpha_rho = std::min(alpha_rho, computeVenkatAlpha(phi_i, phi_pred, rho_min, rho_max, cell_vol));

            phi_i = cell.rho_u;
            phi_pred = phi_i + cell.rho_u_x * distance_x + cell.rho_u_y * distance_y;
            // alpha = std::min(alpha, computeBarthJespersenAlpha(phi_i, phi_pred, rho_u_min, rho_u_max));
            // alpha_rho_u = std::min(alpha_rho_u, computeBarthJespersenAlpha(phi_i, phi_pred, rho_u_min, rho_u_max));
            alpha_rho_u = std::min(alpha_rho_u, computeVenkatAlpha(phi_i, phi_pred, rho_u_min, rho_u_max, cell_vol));

            phi_i = cell.rho_v;
            phi_pred = phi_i + cell.rho_v_x * distance_x + cell.rho_v_y * distance_y;
            // alpha = std::min(alpha, computeBarthJespersenAlpha(phi_i, phi_pred, rho_v_min, rho_v_max));
            // alpha_rho_v = std::min(alpha_rho_v, computeBarthJespersenAlpha(phi_i, phi_pred, rho_v_min, rho_v_max));
            alpha_rho_v = std::min(alpha_rho_v, computeVenkatAlpha(phi_i, phi_pred, rho_v_min, rho_v_max, cell_vol));

            phi_i = cell.E;
            phi_pred = phi_i + cell.E_x * distance_x + cell.E_y * distance_y;
            // alpha = std::min(alpha, computeBarthJespersenAlpha(phi_i, phi_pred, E_min, E_max));
            // alpha_E = std::min(alpha_E, computeBarthJespersenAlpha(phi_i, phi_pred, E_min, E_max));
            alpha_E = std::min(alpha_E, computeVenkatAlpha(phi_i, phi_pred, E_min, E_max, cell_vol));
        }

        for (std::size_t edgeID : cell.edgeIDs) {  
            Edge& edge = mesh.edges[edgeID];
            double edge_x = edge.centre.x, edge_y = edge.centre.y;
            double distance_x = edge_x - cell_x, distance_y = edge_y - cell_y;
            
            // alpha = 1;
            if (edge.leftCellID == cid) {
                edge.rho_L   = cell.rho   + alpha_rho * (cell.rho_x * distance_x + cell.rho_y * distance_y);
                edge.rho_u_L = cell.rho_u + alpha_rho_u * (cell.rho_u_x * distance_x + cell.rho_u_y * distance_y);
                edge.rho_v_L = cell.rho_v + alpha_rho_v * (cell.rho_v_x * distance_x + cell.rho_v_y * distance_y);
                edge.E_L     = cell.E     + alpha_E * (cell.E_x * distance_x + cell.E_y * distance_y);
                if (edge.boundaryCondition == 'i' || edge.boundaryCondition == 'o' || edge.boundaryCondition == 'f' || edge.boundaryCondition == 'w') {
                    double rho = edge.rho_L, rho_u = edge.rho_u_L, rho_v = edge.rho_v_L, E = edge.E_L;
                    apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge.rho_L, edge.rho_u_L, edge.rho_v_L, edge.E_L, edge.rho_R, edge.rho_u_R, edge.rho_v_R, edge.E_R, edge.edgeNormal.x, edge.edgeNormal.y);
                }
            }
            else if (edge.rightCellID == cid) {
                edge.rho_R   = cell.rho   + alpha_rho * (cell.rho_x * distance_x + cell.rho_y * distance_y);
                edge.rho_u_R = cell.rho_u + alpha_rho_u * (cell.rho_u_x * distance_x + cell.rho_u_y * distance_y);
                edge.rho_v_R = cell.rho_v + alpha_rho_v * (cell.rho_v_x * distance_x + cell.rho_v_y * distance_y);
                edge.E_R     = cell.E     + alpha_E * (cell.E_x * distance_x + cell.E_y * distance_y);
            }
        }
    }
    time_toc(t_inter, "Interface states", params.verbosity); 
}



inline double pressure(double rho, double u, double v, double E, double gamma){
    double kinetic = 0.5*rho*(u*u + v*v);
    return (gamma - 1.0)*(E - kinetic);
}

inline double sound_speed(double gamma, double p, double rho) {
    return std::sqrt(gamma * p / rho);
}


void compute_hllc_flux(Edge& edge, const SimulationParams& params, double& edge_flux_rho, double& edge_flux_rho_u, double& edge_flux_rho_v, double& edge_flux_E) {
    
    double nx = -edge.edgeNormal.x, ny = -edge.edgeNormal.y;

    // 1) Variables primitives
    double rho_L = edge.rho_L;
    double u_L   = (edge.rho_u_L*( nx) + edge.rho_v_L*ny)/rho_L;
    double v_L   = (edge.rho_u_L*(-ny) + edge.rho_v_L*nx)/rho_L;
    double E_L   = edge.E_L;
    double p_L   = pressure(edge.rho_L, u_L, v_L, E_L, params.gamma);
    double c_L   = sound_speed(params.gamma, p_L, edge.rho_L);

    double rho_R = edge.rho_R;
    double u_R   = (edge.rho_u_R*( nx) + edge.rho_v_R*ny)/rho_R;
    double v_R   = (edge.rho_u_R*(-ny) + edge.rho_v_R*nx)/rho_R;
    double E_R   = edge.E_R;
    double p_R   = pressure(rho_R, u_R, v_R, E_R, params.gamma);
    double c_R   = sound_speed(params.gamma, p_R, rho_R);
    
    // 2) Estimer les vitesses S_L et S_R
    double S_L = std::min(u_L - c_L, u_R - c_R);
    double S_R = std::max(u_L + c_L, u_R + c_R);

    // 3) Vitesse de contact
    double numer = p_R - p_L + rho_L*u_L*(S_L - u_L) - rho_R*u_R*(S_R - u_R);
    double denom = rho_L*(S_L - u_L) - rho_R*(S_R - u_R);
    double S_star = numer / denom;

    // 4) Etats intermediaires

    // densites
    double rho_L_star = rho_L * (S_L - u_L) / (S_L - S_star);
    double rho_R_star = rho_R * (S_R - u_R) / (S_R - S_star);

    // conserve v
    double rho_v_L_star = rho_L_star * v_L;
    double rho_v_R_star = rho_R_star * v_R;
    
    // pression dans la region star (identique a gauche ou a droite)
    double p_star = p_L + rho_L*(S_L - u_L)*(S_star - u_L);

    // energie totale dans chaque sous-region
    double E_L_star = ((S_L - u_L)*E_L - p_L*u_L + p_star*S_star) / (S_L - S_star);
    double E_R_star = ((S_R - u_R)*E_R - p_R*u_R + p_star*S_star) / (S_R - S_star);
    
    double rho_u_L_star = rho_L_star * S_star;
    double rho_u_R_star = rho_R_star * S_star;
    
    double flux_rho_u, flux_rho_v;

    if (S_L >= 0.0){
        edge_flux_rho   = rho_L * u_L; // edge.rho_u_L;
        flux_rho_u = rho_L * u_L * u_L + p_L; // edge.rho_u_L;
        flux_rho_v = rho_L * u_L * v_L; // edge.rho_u_L;
        edge_flux_E     = (E_L + p_L) * u_L;
    }
    else if (S_L <= 0.0 && S_star >= 0.0){
        edge_flux_rho   = rho_u_L_star;
        flux_rho_u = rho_u_L_star * rho_u_L_star / rho_L_star + p_star;
        flux_rho_v = rho_v_L_star * rho_u_L_star / rho_L_star;
        edge_flux_E     = (E_L_star + p_star) * rho_u_L_star / rho_L_star;
    }
    else if (S_star <= 0.0 && S_R >= 0.0){
        edge_flux_rho   = rho_u_R_star;
        flux_rho_u = rho_u_R_star * rho_u_R_star / rho_R_star + p_star;
        flux_rho_v = rho_v_R_star * rho_u_R_star / rho_R_star;
        edge_flux_E     = (E_R_star + p_star) * rho_u_R_star / rho_R_star;
    }
    else{  // S_R <= 0
        edge_flux_rho   = rho_R * u_R; // edge.rho_u_R;
        flux_rho_u = rho_R * u_R * u_R + p_R; // edge.rho_u_R;
        flux_rho_v = rho_R * u_R * v_R; // edge.rho_u_R;
        edge_flux_E     = (E_R + p_R) * u_R;
    }

    edge_flux_rho_u = (flux_rho_u*nx + flux_rho_v*(-ny));
    edge_flux_rho_v = (flux_rho_u*ny + flux_rho_v*( nx));
}


void computeAdvectionTerm(Mesh& mesh, const SimulationParams& params) {


    computeScalarGradientAdvection(mesh, params);
    computeInterfaceStates(mesh, params);


    static std::vector<double> edges_flux_rho, edges_flux_rho_u, edges_flux_rho_v, edges_flux_E;
    bool first = edges_flux_rho.empty();
    if (first) {
        size_t N_edges = mesh.edges.size();
        edges_flux_rho.resize(N_edges);
        edges_flux_rho_u.resize(N_edges);
        edges_flux_rho_v.resize(N_edges);
        edges_flux_E.resize(N_edges);
    }

    auto t_hllc = time_tic();
    #pragma omp parallel for 
    for (std::size_t i = 0; i < mesh.edges.size(); ++i) {
        Edge& edge = mesh.edges[i]; 
        compute_hllc_flux(edge, params, edges_flux_rho[i], edges_flux_rho_u[i], edges_flux_rho_v[i], edges_flux_E[i]);
    }

    #pragma omp parallel for 
    for(std::size_t cid=0; cid<mesh.cells.size(); ++cid){
        Cell& cell = mesh.cells[cid];
        
        cell.R_rho   = 0.0;
        cell.R_rho_u = 0.0;
        cell.R_rho_v = 0.0;
        cell.R_E     = 0.0;

        for(std::size_t eid : cell.edgeIDs){
            const Edge& edge = mesh.edges[eid];
            double sign = (edge.leftCellID == cid ? +1.0 : -1.0);
            double L = edge.edgeLength;

            cell.R_rho   += sign * edges_flux_rho[eid]   * L;
            cell.R_rho_u += sign * edges_flux_rho_u[eid] * L;
            cell.R_rho_v += sign * edges_flux_rho_v[eid] * L;
            cell.R_E     += sign * edges_flux_E[eid]     * L;
        }

        double invV = 1.0 / cell.cellVolume;
        cell.R_rho   *= -invV;
        cell.R_rho_u *= -invV;
        cell.R_rho_v *= -invV;
        cell.R_E     *= -invV;
    }
    time_toc(t_hllc, "HLLC", params.verbosity); 
}









} // namespace navier_stokes






// void computeAdvectionTerm(Mesh& mesh, const SimulationParams& params) { 

//     #pragma omp parallel for
//     for (std::size_t e = 0; e < mesh.edges.size(); ++e) {
//         auto& edge = mesh.edges[e]; 
//         double nx = edge.edgeNormal.x;
//         double ny = edge.edgeNormal.y;
 
//         double rhoL = edge.rho_L;
//         double uL_n = (edge.rho_u_L * nx + edge.rho_v_L * ny) / rhoL;
//         double rhoR = edge.rho_R;
//         double uR_n = (edge.rho_u_R * nx + edge.rho_v_R * ny) / rhoR;

//         // Pressions gauche/droite
//         double pL = pressure(rhoL, 
//                              (edge.rho_u_L*nx + edge.rho_v_L*(-ny))/rhoL,
//                              (edge.rho_u_L*ny + edge.rho_v_L*nx)/rhoL,
//                              edge.E_L, params.gamma);
//         double pR = pressure(rhoR,
//                              (edge.rho_u_R*nx + edge.rho_v_R*(-ny))/rhoR,
//                              (edge.rho_u_R*ny + edge.rho_v_R*nx)/rhoR,
//                              edge.E_R, params.gamma);
 
//         double F_rho, F_mom_n, F_mom_t, F_E;
//         if (uL_n >= 0) {
//             F_rho    = rhoL * uL_n;
//             F_mom_n  = rhoL * uL_n * uL_n + pL;
//             F_mom_t  = rhoL * uL_n * 0.0;  
//             F_E      = (edge.E_L + pL) * uL_n;
//         } else {
//             F_rho    = rhoR * uR_n;
//             F_mom_n  = rhoR * uR_n * uR_n + pR;
//             F_mom_t  = rhoR * uR_n * 0.0;
//             F_E      = (edge.E_R + pR) * uR_n;
//         }

//         // 4. Projection vers x/y
//         edge.flux_rho   = F_rho;
//         edge.flux_rho_u = F_mom_n * nx - F_mom_t * ny;
//         edge.flux_rho_v = F_mom_n * ny + F_mom_t * nx;
//         edge.flux_E     = F_E;
//     }
 
//     #pragma omp parallel for
//     for (std::size_t cid = 0; cid < mesh.cells.size(); ++cid) {
//         auto& cell = mesh.cells[cid];
//         cell.R_rho   = 0.0;
//         cell.R_rho_u = 0.0;
//         cell.R_rho_v = 0.0;
//         cell.R_E     = 0.0;

//         for (auto eid : cell.edgeIDs) {
//             const auto& e = mesh.edges[eid]; 
//             double sign = (e.leftCellID == cid ? -1.0 : +1.0);
//             // double sign = (e.leftCellID == cid ? +1.0 : -1.0);
//             double L    = e.edgeLength;

//             cell.R_rho   += sign * e.flux_rho   * L;
//             cell.R_rho_u += sign * e.flux_rho_u * L;
//             cell.R_rho_v += sign * e.flux_rho_v * L;
//             cell.R_E     += sign * e.flux_E     * L;
//         }

//         double invV = 1.0 / cell.cellVolume; 
//         cell.R_rho   *= -invV;
//         cell.R_rho_u *= -invV;
//         cell.R_rho_v *= -invV;
//         cell.R_E     *= -invV;
//     }
// }





    // std::cout << "vertices_rho[1] " << vertices_rho[1] << " vertices_rho_u[1] " << vertices_rho_u[1] 
    //           << " vertices_rho_v[1] " << vertices_rho_v[1] << " vertices_E[1] " << vertices_E[1] << '\n' << std::flush;

    // std::cout << "vertices_rho[208] " << vertices_rho[208] << " vertices_rho_u[208] " << vertices_rho_u[208] 
    //           << " vertices_rho_v[208] " << vertices_rho_v[208] << " vertices_E[208] " << vertices_E[208] << '\n' << std::flush;

    // vertices_rho[1]   = vertices_rho[208];
    // vertices_rho_u[1] = vertices_rho_u[208];
    // vertices_rho_v[1] = vertices_rho_v[208];
    // vertices_E[1]     = vertices_E[208];



    // // mesh.cells[21413].rho   = mesh.cells[21414].rho;
    // // mesh.cells[21413].rho_u = mesh.cells[21414].rho_u;
    // // mesh.cells[21413].rho_v = mesh.cells[21414].rho_v;
    // // mesh.cells[21413].E     = mesh.cells[21414].E;

    // // mesh.cells[25196].rho   = mesh.cells[25157].rho;
    // // mesh.cells[25196].rho_u = mesh.cells[25157].rho_u;
    // // mesh.cells[25196].rho_v = mesh.cells[25157].rho_v;
    // // mesh.cells[25196].E     = mesh.cells[25157].E;

    // mesh.cells[5184].rho   = mesh.cells[5160].rho;
    // mesh.cells[5184].rho_u = mesh.cells[5160].rho_u;
    // mesh.cells[5184].rho_v = mesh.cells[5160].rho_v;
    // mesh.cells[5184].E     = mesh.cells[5160].E;
    // mesh.cells[5208].rho   = mesh.cells[5184].rho;
    // mesh.cells[5208].rho_u = mesh.cells[5184].rho_u;
    // mesh.cells[5208].rho_v = mesh.cells[5184].rho_v;
    // mesh.cells[5208].E     = mesh.cells[5184].E;
    // mesh.cells[5232].rho   = mesh.cells[5208].rho;
    // mesh.cells[5232].rho_u = mesh.cells[5208].rho_u;
    // mesh.cells[5232].rho_v = mesh.cells[5208].rho_v;
    // mesh.cells[5232].E     = mesh.cells[5208].E;

    // mesh.cells[2831].rho   = mesh.cells[2830].rho;
    // mesh.cells[2831].rho_u = mesh.cells[2830].rho_u;
    // mesh.cells[2831].rho_v = mesh.cells[2830].rho_v;
    // mesh.cells[2831].E     = mesh.cells[2830].E;
    // mesh.cells[6101].rho   = mesh.cells[2831].rho;
    // mesh.cells[6101].rho_u = mesh.cells[2831].rho_u;
    // mesh.cells[6101].rho_v = mesh.cells[2831].rho_v;
    // mesh.cells[6101].E     = mesh.cells[2831].E;
    // mesh.cells[6102].rho   = mesh.cells[6101].rho;
    // mesh.cells[6102].rho_u = mesh.cells[6101].rho_u;
    // mesh.cells[6102].rho_v = mesh.cells[6101].rho_v;
    // mesh.cells[6102].E     = mesh.cells[6101].E;









