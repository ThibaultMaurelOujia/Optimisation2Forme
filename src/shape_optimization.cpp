#include "shape_optimization.hpp"


namespace navier_stokes {













/// 0) Lecture/écriture

// trim
static inline void ltrim(std::string& s){ s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c){return !std::isspace(c);})); }
static inline void rtrim(std::string& s){ s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c){return !std::isspace(c);}).base(), s.end()); }
static inline void trim(std::string& s){ ltrim(s); rtrim(s); }

// Parse un fichier "clé (=|*=|espace) valeur"
// - Remplit param_names_all, param_values_all 
// - Remplit param_names, uniquement les clés marquées "*="
void loadParamsGeneric(const std::string& filepath,
                       std::vector<std::string>& param_names_all,  
                       std::vector<double>& param_values_all, 
                       std::vector<std::string>&  param_names)       //(opt)
{
    param_names_all.clear();
    param_values_all.clear();
    param_names.clear();

    if (filepath.empty() || filepath == "none") return;

    std::ifstream in(filepath);
    if (!in) {
        throw std::runtime_error("Impossible d'ouvrir le fichier de paramètres: " + filepath);
    }

    // position/index de chaque clé déjà vue dans param_names_all
    std::unordered_map<std::string, std::size_t> pos_of_key;
    // ensemble des clés marquées à optimisées
    std::unordered_set<std::string> is_opt;

    std::string line;
    while (std::getline(in, line)) {
        // retirer commentaires
        if (auto p = line.find('#'); p != std::string::npos) line.erase(p);
        trim(line);
        if (line.empty()) continue;

        bool mark_opt = false; // cette ligne contient "*=" ?
        std::string key, sval;

        // priorité à "*="
        if (auto k = line.find("*="); k != std::string::npos) {
            mark_opt = true;
            key  = line.substr(0, k);
            sval = line.substr(k + 2);
            trim(key); trim(sval);
        } else if (auto e = line.find('='); e != std::string::npos) {
            key  = line.substr(0, e);
            sval = line.substr(e + 1);
            trim(key); trim(sval);
        } else {
            std::istringstream ss(line);
            ss >> key >> sval;
            trim(key); trim(sval);
        }

        if (key.empty() || sval.empty()) continue;

        // conversion
        double v = std::stod(sval);

        auto it = pos_of_key.find(key);
        if (it == pos_of_key.end()) {
            // première occurrence de cette clé
            std::size_t idx = param_names_all.size();
            pos_of_key[key] = idx;
            param_names_all.push_back(key);
            param_values_all.push_back(v);
        } else {
            // clé déjà existante -> on remplace la valeur
            param_values_all[it->second] = v;
        }

        // statut "optimisé"  
        if (mark_opt) {
            is_opt.insert(key);
        } else {
            // si plus d'astérisque, on retire de la liste optimisée
            is_opt.erase(key);
        }
    }

    // construire param_names dans l'ordre d'apparition
    param_names.reserve(is_opt.size());
    for (const auto& k : param_names_all) {
        if (is_opt.find(k) != is_opt.end())
            param_names.push_back(k);
    }
}


double getParamOrDefault(const std::vector<std::string>& names,
                         const std::vector<double>& values,
                         const std::string& key,
                         double def_value){
    auto it = std::find(names.begin(), names.end(), key);
    if (it == names.end()) return def_value;
    std::size_t idx = static_cast<std::size_t>(std::distance(names.begin(), it));
    return values[idx];
}



void appendShapeOptLog(const SimulationParams& params, const std::vector<std::string>& param_names,
                       double J_current, const std::vector<double>& grad) {
    const std::string& dir = params.shape_log_dir;
    if (dir.empty() || dir == "none") return;

    std::filesystem::create_directories(dir);
    const std::string path = (std::filesystem::path(dir) / "shape_opt.log").string();

    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Impossible d'ouvrir le fichier log: " + path);
    }

    out << std::fixed << std::setprecision(8);
    out << " J=" << J_current;

    // Gradients compacts: " m=-2.1635e+05 p=+1.23e+03 "
    out << std::scientific << std::setprecision(6);
    for (std::size_t k = 0; k < param_names.size(); ++k) {
        out << ' ' << param_names[k] << '=' << grad[k];
    }
    out << '\n';
}























/// 1) computeResidual
// !!! pas de terme visqueux !!!
void computeResidual(Mesh& mesh, const SimulationParams& params, 
                    std::vector<double>& R_rho, std::vector<double>& R_rho_u, std::vector<double>& R_rho_v, std::vector<double>& R_E) {
    auto t_R = time_tic();

    
    computeAdvectionTerm(mesh, params);
    computeDiffusionTerm(mesh, params);
     
    #pragma omp parallel for 
    for (size_t i = 0; i < mesh.cells.size(); ++i) {
        R_rho[i]   = mesh.cells[i].R_rho;
        R_rho_u[i] = mesh.cells[i].R_rho_u;
        R_rho_v[i] = mesh.cells[i].R_rho_v;
        R_E[i]     = mesh.cells[i].R_E;
    }
    
    time_toc(t_R, "computeResidual", params.verbosity); 

}

/// 2) applyJacobian
void applyJacobian(Mesh mesh, const SimulationParams& params, // Différence centrée
                   const std::vector<double>& x_rho, const std::vector<double>& x_rho_u, const std::vector<double>& x_rho_v, const std::vector<double>& x_E, 
                   std::vector<double>& y_rho, std::vector<double>& y_rho_u, std::vector<double>& y_rho_v, std::vector<double>& y_E) {

    const double eps = 1e-6;
    
    // Différence centrée
    for (int i = 0; i < mesh.cells.size(); ++i) {
        mesh.cells[i].rho   -= eps * x_rho[i];
        mesh.cells[i].rho_u -= eps * x_rho_u[i];
        mesh.cells[i].rho_v -= eps * x_rho_v[i];
        mesh.cells[i].E     -= eps * x_E[i];
    }
    
    std::vector<double> R0_rho(mesh.cells.size()), R0_rho_u(mesh.cells.size()), R0_rho_v(mesh.cells.size()), R0_E(mesh.cells.size());
    computeResidual(mesh, params, R0_rho, R0_rho_u, R0_rho_v, R0_E); 
    
    // Différence centrée
    for (int i = 0; i < mesh.cells.size(); ++i) {
        mesh.cells[i].rho   += 2*eps * x_rho[i];
        mesh.cells[i].rho_u += 2*eps * x_rho_u[i];
        mesh.cells[i].rho_v += 2*eps * x_rho_v[i];
        mesh.cells[i].E     += 2*eps * x_E[i];
    }

    std::vector<double> R1_rho(mesh.cells.size()), R1_rho_u(mesh.cells.size()), R1_rho_v(mesh.cells.size()), R1_E(mesh.cells.size());
    computeResidual(mesh, params, R1_rho, R1_rho_u, R1_rho_v, R1_E); 

    // Différence centrée
    for (size_t i = 0; i < mesh.cells.size(); ++i) {
        y_rho[i]   = (R1_rho[i] - R0_rho[i]) / (2*eps);
        y_rho_u[i] = (R1_rho_u[i] - R0_rho_u[i]) / (2*eps);
        y_rho_v[i] = (R1_rho_v[i] - R0_rho_v[i]) / (2*eps);
        y_E[i]     = (R1_E[i] - R0_E[i]) / (2*eps);
    }
}


/// 3) applyJacobianTranspose

struct EdgeFluxJac {
    double F_UL[4][4]; // dF/du_L
    double F_UR[4][4]; // dF/du_L
};

void applyJacobianTranspose(Mesh& mesh, const SimulationParams& params,
                   const std::vector<double>& x_rho, const std::vector<double>& x_rho_u, const std::vector<double>& x_rho_v, const std::vector<double>& x_E, 
                   std::vector<double>& y_rho, std::vector<double>& y_rho_u, std::vector<double>& y_rho_v, std::vector<double>& y_E, 
                   double eps = 1e-6) {

    //const double eps = 1e-6;
    // !!!!!!!!!!!!!!! remplacer eps fixe par un pas relatif par composante (et clamp rho et E > 0) pour plus de robustesse numérique !!!!!!!!!!!!!!!

    #pragma omp parallel for 
    for (std::size_t i = 0; i < mesh.edges.size(); ++i) {
        const Edge& edge = mesh.edges[i]; 
        const double L = edge.edgeLength;
        
        // EdgeFluxJac& J_edge = J_edges[i];
        EdgeFluxJac J_edge;

        double edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p;
        double edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m;

        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                J_edge.F_UL[r][c] = 0.0;
                J_edge.F_UR[r][c] = 0.0;
            }
        }


        // L
        {
            Edge edge_plus = edge, edge_minus = edge;
            edge_plus.rho_L  = edge.rho_L + eps;
            edge_minus.rho_L = edge.rho_L - eps;

            if (edge.boundaryCondition != '\0') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_L, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_L, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R);
            }

            compute_hllc_flux(edge_plus, params, edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p);
            compute_hllc_flux(edge_minus, params, edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m);

            J_edge.F_UL[0][0] = (edge_flux_rho_p - edge_flux_rho_m) / (2*eps);
            J_edge.F_UL[1][0] = (edge_flux_rho_u_p - edge_flux_rho_u_m) / (2*eps);
            J_edge.F_UL[2][0] = (edge_flux_rho_v_p - edge_flux_rho_v_m) / (2*eps);
            J_edge.F_UL[3][0] = (edge_flux_E_p - edge_flux_E_m) / (2*eps);
        }

        {
            Edge edge_plus = edge, edge_minus = edge;
            edge_plus.rho_u_L  = edge.rho_u_L + eps;
            edge_minus.rho_u_L = edge.rho_u_L - eps;

            if (edge.boundaryCondition != '\0') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_L, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_L, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R);
            }

            compute_hllc_flux(edge_plus, params, edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p);
            compute_hllc_flux(edge_minus, params, edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m);

            J_edge.F_UL[0][1] = (edge_flux_rho_p - edge_flux_rho_m) / (2*eps);
            J_edge.F_UL[1][1] = (edge_flux_rho_u_p - edge_flux_rho_u_m) / (2*eps);
            J_edge.F_UL[2][1] = (edge_flux_rho_v_p - edge_flux_rho_v_m) / (2*eps);
            J_edge.F_UL[3][1] = (edge_flux_E_p - edge_flux_E_m) / (2*eps);
        }

        {
            Edge edge_plus = edge, edge_minus = edge;
            edge_plus.rho_v_L  = edge.rho_v_L + eps;
            edge_minus.rho_v_L = edge.rho_v_L - eps;

            if (edge.boundaryCondition != '\0') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_L, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_L, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R);
            }

            compute_hllc_flux(edge_plus, params, edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p);
            compute_hllc_flux(edge_minus, params, edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m);

            J_edge.F_UL[0][2] = (edge_flux_rho_p - edge_flux_rho_m) / (2*eps);
            J_edge.F_UL[1][2] = (edge_flux_rho_u_p - edge_flux_rho_u_m) / (2*eps);
            J_edge.F_UL[2][2] = (edge_flux_rho_v_p - edge_flux_rho_v_m) / (2*eps);
            J_edge.F_UL[3][2] = (edge_flux_E_p - edge_flux_E_m) / (2*eps);
        }

        {
            Edge edge_plus = edge, edge_minus = edge;
            edge_plus.E_L  = edge.E_L + eps;
            edge_minus.E_L = edge.E_L - eps;

            if (edge.boundaryCondition != '\0') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_L, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_L, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R);
            }

            compute_hllc_flux(edge_plus, params, edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p);
            compute_hllc_flux(edge_minus, params, edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m);

            J_edge.F_UL[0][3] = (edge_flux_rho_p - edge_flux_rho_m) / (2*eps);
            J_edge.F_UL[1][3] = (edge_flux_rho_u_p - edge_flux_rho_u_m) / (2*eps);
            J_edge.F_UL[2][3] = (edge_flux_rho_v_p - edge_flux_rho_v_m) / (2*eps);
            J_edge.F_UL[3][3] = (edge_flux_E_p - edge_flux_E_m) / (2*eps);
        }


        // R
        {
            Edge edge_plus = edge, edge_minus = edge;
            edge_plus.rho_R  = edge.rho_R + eps;
            edge_minus.rho_R = edge.rho_R - eps;

            if (edge.boundaryCondition == 'w') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_R);
            } else if (edge.boundaryCondition != '\0') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_L, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_L, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R);
            }

            compute_hllc_flux(edge_plus, params, edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p);
            compute_hllc_flux(edge_minus, params, edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m);

            J_edge.F_UR[0][0] = (edge_flux_rho_p - edge_flux_rho_m) / (2*eps);
            J_edge.F_UR[1][0] = (edge_flux_rho_u_p - edge_flux_rho_u_m) / (2*eps);
            J_edge.F_UR[2][0] = (edge_flux_rho_v_p - edge_flux_rho_v_m) / (2*eps);
            J_edge.F_UR[3][0] = (edge_flux_E_p - edge_flux_E_m) / (2*eps);
        }

        {
            Edge edge_plus = edge, edge_minus = edge;
            edge_plus.rho_u_R  = edge.rho_u_R + eps;
            edge_minus.rho_u_R = edge.rho_u_R - eps;

            if (edge.boundaryCondition == 'w') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_R);
            } else if (edge.boundaryCondition != '\0') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_L, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_L, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R);
            }

            compute_hllc_flux(edge_plus, params, edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p);
            compute_hllc_flux(edge_minus, params, edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m);

            J_edge.F_UR[0][1] = (edge_flux_rho_p - edge_flux_rho_m) / (2*eps);
            J_edge.F_UR[1][1] = (edge_flux_rho_u_p - edge_flux_rho_u_m) / (2*eps);
            J_edge.F_UR[2][1] = (edge_flux_rho_v_p - edge_flux_rho_v_m) / (2*eps);
            J_edge.F_UR[3][1] = (edge_flux_E_p - edge_flux_E_m) / (2*eps);
        }

        {
            Edge edge_plus = edge, edge_minus = edge;
            edge_plus.rho_v_R  = edge.rho_v_R + eps;
            edge_minus.rho_v_R = edge.rho_v_R - eps;

            if (edge.boundaryCondition == 'w') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_R);
            } else if (edge.boundaryCondition != '\0') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_L, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_L, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R);
            }

            compute_hllc_flux(edge_plus, params, edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p);
            compute_hllc_flux(edge_minus, params, edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m);

            J_edge.F_UR[0][2] = (edge_flux_rho_p - edge_flux_rho_m) / (2*eps);
            J_edge.F_UR[1][2] = (edge_flux_rho_u_p - edge_flux_rho_u_m) / (2*eps);
            J_edge.F_UR[2][2] = (edge_flux_rho_v_p - edge_flux_rho_v_m) / (2*eps);
            J_edge.F_UR[3][2] = (edge_flux_E_p - edge_flux_E_m) / (2*eps);
        }

        {
            Edge edge_plus = edge, edge_minus = edge;
            edge_plus.E_R  = edge.E_R + eps;
            edge_minus.E_R = edge.E_R - eps;

            if (edge.boundaryCondition == 'w') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_R);
            } else if (edge.boundaryCondition != '\0') {
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_plus.rho_L, edge_plus.rho_u_L, edge_plus.rho_v_L, edge_plus.E_L, edge_plus.rho_R, edge_plus.rho_u_R, edge_plus.rho_v_R, edge_plus.E_R);
                apply_InterfaceStates_boundary_conditions(params, edge.boundaryCondition, edge_minus.rho_L, edge_minus.rho_u_L, edge_minus.rho_v_L, edge_minus.E_L, edge_minus.rho_R, edge_minus.rho_u_R, edge_minus.rho_v_R, edge_minus.E_R);
            }

            compute_hllc_flux(edge_plus, params, edge_flux_rho_p, edge_flux_rho_u_p, edge_flux_rho_v_p, edge_flux_E_p);
            compute_hllc_flux(edge_minus, params, edge_flux_rho_m, edge_flux_rho_u_m, edge_flux_rho_v_m, edge_flux_E_m);

            J_edge.F_UR[0][3] = (edge_flux_rho_p - edge_flux_rho_m) / (2*eps);
            J_edge.F_UR[1][3] = (edge_flux_rho_u_p - edge_flux_rho_u_m) / (2*eps);
            J_edge.F_UR[2][3] = (edge_flux_rho_v_p - edge_flux_rho_v_m) / (2*eps);
            J_edge.F_UR[3][3] = (edge_flux_E_p - edge_flux_E_m) / (2*eps);
        }



        const std::size_t iL = edge.leftCellID;
        Cell& cell_L = mesh.cells[iL];
        double invV_L = 1.0 / cell_L.cellVolume;

        if (edge.boundaryCondition != '\0') {
            double Delta_x_rho   = (- x_rho[iL]   * invV_L) * L;
            double Delta_x_rho_u = (- x_rho_u[iL] * invV_L) * L;
            double Delta_x_rho_v = (- x_rho_v[iL] * invV_L) * L;
            double Delta_x_E     = (- x_E[iL]     * invV_L) * L;

            #pragma omp atomic
            y_rho[iL]   += (J_edge.F_UL[0][0] * Delta_x_rho + J_edge.F_UL[1][0] * Delta_x_rho_u + J_edge.F_UL[2][0] * Delta_x_rho_v + J_edge.F_UL[3][0] * Delta_x_E);
            #pragma omp atomic
            y_rho_u[iL] += (J_edge.F_UL[0][1] * Delta_x_rho + J_edge.F_UL[1][1] * Delta_x_rho_u + J_edge.F_UL[2][1] * Delta_x_rho_v + J_edge.F_UL[3][1] * Delta_x_E);
            #pragma omp atomic
            y_rho_v[iL] += (J_edge.F_UL[0][2] * Delta_x_rho + J_edge.F_UL[1][2] * Delta_x_rho_u + J_edge.F_UL[2][2] * Delta_x_rho_v + J_edge.F_UL[3][2] * Delta_x_E);
            #pragma omp atomic
            y_E[iL]     += (J_edge.F_UL[0][3] * Delta_x_rho + J_edge.F_UL[1][3] * Delta_x_rho_u + J_edge.F_UL[2][3] * Delta_x_rho_v + J_edge.F_UL[3][3] * Delta_x_E);
            
            continue;
        }

        const std::size_t iR = edge.rightCellID;
        Cell& cell_R = mesh.cells[iR];
        double invV_R = 1.0 / cell_R.cellVolume;


        double Delta_x_rho   = (- x_rho[iL]   * invV_L + x_rho[iR]   * invV_R) * L;
        double Delta_x_rho_u = (- x_rho_u[iL] * invV_L + x_rho_u[iR] * invV_R) * L;
        double Delta_x_rho_v = (- x_rho_v[iL] * invV_L + x_rho_v[iR] * invV_R) * L;
        double Delta_x_E     = (- x_E[iL]     * invV_L + x_E[iR]     * invV_R) * L;

        #pragma omp atomic
        y_rho[iL]   += (J_edge.F_UL[0][0] * Delta_x_rho + J_edge.F_UL[1][0] * Delta_x_rho_u + J_edge.F_UL[2][0] * Delta_x_rho_v + J_edge.F_UL[3][0] * Delta_x_E);
        #pragma omp atomic
        y_rho_u[iL] += (J_edge.F_UL[0][1] * Delta_x_rho + J_edge.F_UL[1][1] * Delta_x_rho_u + J_edge.F_UL[2][1] * Delta_x_rho_v + J_edge.F_UL[3][1] * Delta_x_E);
        #pragma omp atomic
        y_rho_v[iL] += (J_edge.F_UL[0][2] * Delta_x_rho + J_edge.F_UL[1][2] * Delta_x_rho_u + J_edge.F_UL[2][2] * Delta_x_rho_v + J_edge.F_UL[3][2] * Delta_x_E);
        #pragma omp atomic
        y_E[iL]     += (J_edge.F_UL[0][3] * Delta_x_rho + J_edge.F_UL[1][3] * Delta_x_rho_u + J_edge.F_UL[2][3] * Delta_x_rho_v + J_edge.F_UL[3][3] * Delta_x_E);

        #pragma omp atomic
        y_rho[iR]   += (J_edge.F_UR[0][0] * Delta_x_rho + J_edge.F_UR[1][0] * Delta_x_rho_u + J_edge.F_UR[2][0] * Delta_x_rho_v + J_edge.F_UR[3][0] * Delta_x_E);
        #pragma omp atomic
        y_rho_u[iR] += (J_edge.F_UR[0][1] * Delta_x_rho + J_edge.F_UR[1][1] * Delta_x_rho_u + J_edge.F_UR[2][1] * Delta_x_rho_v + J_edge.F_UR[3][1] * Delta_x_E);
        #pragma omp atomic
        y_rho_v[iR] += (J_edge.F_UR[0][2] * Delta_x_rho + J_edge.F_UR[1][2] * Delta_x_rho_u + J_edge.F_UR[2][2] * Delta_x_rho_v + J_edge.F_UR[3][2] * Delta_x_E);
        #pragma omp atomic
        y_E[iR]     += (J_edge.F_UR[0][3] * Delta_x_rho + J_edge.F_UR[1][3] * Delta_x_rho_u + J_edge.F_UR[2][3] * Delta_x_rho_v + J_edge.F_UR[3][3] * Delta_x_E);

    }
}


/// 4) assembleAdjointRHS
inline double sound_speed_ideal(double gamma, double p, double rho){ return std::sqrt(gamma * p / rho); }

// !!! pas de terme visqueux !!!
void assembleAdjointRHS(Mesh& mesh, const SimulationParams& params,
                        std::vector<double>& b_rho, std::vector<double>& b_rho_u, std::vector<double>& b_rho_v, std::vector<double>& b_E,
                        double beta=1) { // poids de la trainée
    double gamma = params.gamma;
    double rho_inf = params.rho_ref;
    double p_inf   = params.p_ref; 
    const double a_inf   = sound_speed_ideal(gamma,p_inf,rho_inf);
    double M_inf  = params.inflow_velocity;   
    double U_inf = M_inf * a_inf;

    double q_inf = 0.5 * (rho_inf * U_inf*U_inf);
    double corde = 1; // !!! Attention : à adapter

    
    b_rho.assign(mesh.cells.size(), 0.0);
    b_rho_u.assign(mesh.cells.size(), 0.0);
    b_rho_v.assign(mesh.cells.size(), 0.0);
    b_E.assign(mesh.cells.size(), 0.0);


    for (size_t i = 0; i < mesh.edges.size(); ++i) {
        const Edge& edge = mesh.edges[i];
        if (edge.boundaryCondition != 'w') continue;

        size_t iL = edge.leftCellID;
        double A_f = edge.edgeLength;
        double nx  = edge.edgeNormal.x;
        double ny  = edge.edgeNormal.y;

        double rho_L   = edge.rho_L;
        double rho_u_L = edge.rho_u_L;
        double rho_v_L = edge.rho_v_L;
        double E_L     = edge.E_L;

        double dp_drho  = 0.5*(params.gamma-1.0)*((rho_u_L*rho_u_L + rho_v_L*rho_v_L)/(rho_L*rho_L));
        double dp_dru   = -(params.gamma-1.0)*(rho_u_L/rho_L);
        double dp_drv   = -(params.gamma-1.0)*(rho_v_L/rho_L);
        double dp_dE    =  (params.gamma-1.0);

        // b += d(+beta*C_D - C_L)/dU
        
        // b += dC_L/dU_L
        double coeff_CL = +ny * A_f / (q_inf * corde);
        b_rho[iL]   += coeff_CL * dp_drho;
        b_rho_u[iL] += coeff_CL * dp_dru;
        b_rho_v[iL] += coeff_CL * dp_drv;
        b_E[iL]     += coeff_CL * dp_dE;
        
        // b += dC_D/dU_L
        double coeff_CD = -nx * A_f / (q_inf * corde) * beta;
        b_rho[iL]   += coeff_CD * dp_drho;
        b_rho_u[iL] += coeff_CD * dp_dru;
        b_rho_v[iL] += coeff_CD * dp_drv;
        b_E[iL]     += coeff_CD * dp_dE;
    }
}





/// 5) solveAdjointSystem
 
struct JacobianOpT;  // Déclaration anticipée 

}  // fin namespace navier_stokes

// Spécialisation des traits Eigen pour navier_stokes::JacobianOpT  
namespace Eigen { namespace internal {
    template<> struct traits<navier_stokes::JacobianOpT> 
        : public traits<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> 
    {};
} }  // fin de namespace Eigen::internal

namespace navier_stokes {

// Définition de la classe JacobianOpT 
struct JacobianOpT : public Eigen::EigenBase<JacobianOpT> {
    // Types et constantes Eigen requises
    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    enum {
        RowsAtCompileTime    = Eigen::Dynamic,
        ColsAtCompileTime    = Eigen::Dynamic,
        MaxRowsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    // Données membres
    Mesh& mesh;
    const SimulationParams& params;
    Eigen::Index N_glob;
    double eps;

    // Constructeur
    JacobianOpT(Mesh& mesh_, const SimulationParams& params_, Eigen::Index N_glob_, double eps_)
        : mesh(mesh_), params(params_), N_glob(N_glob_), eps(eps_) {}

    // Dimensions de la matrice JacobianOpT
    Eigen::Index rows() const { return N_glob; }
    Eigen::Index cols() const { return N_glob; }

    // Opérateur de multiplication / expression Eigen paresseuse
    template<typename Rhs>
    Eigen::Product<JacobianOpT, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs>& x) const {
        return Eigen::Product<JacobianOpT, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Méthode pour calculer y = J^T * x en utilisant applyJacobianTranspose
    template<typename X, typename Y>
    void _mul(const X& x, Y& y) const {
        const size_t Nc = mesh.cells.size();
        std::vector<double> xr(Nc), xru(Nc), xrv(Nc), xE(Nc);
        for (size_t i = 0; i < Nc; ++i) {
            xr[i]   = x(4*i + 0);
            xru[i]  = x(4*i + 1);
            xrv[i]  = x(4*i + 2);
            xE[i]   = x(4*i + 3);
        }
        std::vector<double> yr(Nc), yru(Nc), yrv(Nc), yE(Nc);
        // applyJacobian(mesh, params, xr, xru, xrv, xE, yr, yru, yrv, yE);
        applyJacobianTranspose(mesh, params, xr, xru, xrv, xE, yr, yru, yrv, yE, eps);
        y.derived().resize(4 * Nc);
        for (size_t i = 0; i < Nc; ++i) {
            y(4*i + 0) = yr[i];
            y(4*i + 1) = yru[i];
            y(4*i + 2) = yrv[i];
            y(4*i + 3) = yE[i];
        }
    }
};

}  // fin namespace navier_stokes

// 4) Produit matrix-free pour JacobianOpT * vecteur 
namespace Eigen { namespace internal {
    template<typename Rhs>
    struct generic_product_impl<navier_stokes::JacobianOpT, Rhs, DenseShape, DenseShape, GemvProduct>
      : generic_product_impl_base<navier_stokes::JacobianOpT, Rhs, generic_product_impl<navier_stokes::JacobianOpT, Rhs>> 
    {
        typedef typename Product<navier_stokes::JacobianOpT, Rhs>::Scalar Scalar;
        template<typename Dest>
        static void scaleAndAddTo(Dest& dst, const navier_stokes::JacobianOpT& lhs, const Rhs& rhs, const Scalar& alpha) {
            assert(alpha == Scalar(1) && "Scaling by a factor != 1 not implemented");
            EIGEN_ONLY_USED_FOR_DEBUG(alpha);
            // Calcul de dst += lhs * rhs
            Eigen::VectorXd tmp(lhs.rows());
            lhs._mul(rhs, tmp);    // effectue le produit JacobianOpT * rhs
            dst += tmp;            // ajoute le résultat au vecteur de destination
        }
    };
} }  // fin de namespace Eigen::internal

namespace navier_stokes {

// Définition de solveAdjointSystem qui utilise JacobianOpT 
void solveAdjointSystem(Mesh& mesh, const SimulationParams& params, std::vector<double>& lambda, double beta=1, double eps = 1e-6) {
    using Eigen::VectorXd;
    const size_t N_cells = mesh.cells.size();
    const size_t N_glob  = 4 * N_cells;

    // assemblage le second membre global b_glob
    std::vector<double> br(N_cells), bru(N_cells), brv(N_cells), bE(N_cells);
    assembleAdjointRHS(mesh, params, br, bru, brv, bE, beta);
    VectorXd b_glob(N_glob);
    for (size_t i = 0; i < N_cells; ++i) {
        b_glob(4*i + 0) = br[i];
        b_glob(4*i + 1) = bru[i];
        b_glob(4*i + 2) = brv[i];
        b_glob(4*i + 3) = bE[i];
    }

    // construction l'opérateur Jacobien matrix-free
    JacobianOpT Jop(mesh, params, static_cast<Eigen::Index>(N_glob), eps);

    // // Résoue J^T * u = b_glob avec BiCGSTAB
    // Eigen::BiCGSTAB<JacobianOpT, Eigen::IdentityPreconditioner> solver;
    // solver.setMaxIterations(10000);
    // solver.setTolerance(1e-4);
    // solver.compute(Jop);                  // Initialise le solveur avec notre opérateur
    // VectorXd u = solver.solve(b_glob);    // Résout Jop * u = b_glob

    // std::cout << "[adjoint] iters=" << solver.iterations() << "  relres=" << solver.error() << "\n";
    // if (solver.info() != Eigen::Success) {
    //     throw std::runtime_error("BiCGSTAB did not converge");
    // }





    // Paramètres
    const int    chunk     = 500;      // log toutes les 500 iters
    const int    max_total = 100000;    // borne supérieure globale
    const double tol       = 1e-6;

    // Préparation solveur
    Eigen::BiCGSTAB<JacobianOpT, Eigen::IdentityPreconditioner> solver;
    solver.setTolerance(tol);
    solver.setMaxIterations(chunk);
    solver.compute(Jop);

    // Inits
    Eigen::VectorXd u = Eigen::VectorXd::Zero(N_glob);   // guess initial
    const double    bnorm = b_glob.norm();
    double          relres = std::numeric_limits<double>::infinity();
    int             it_total = 0;

    while (it_total < max_total) {
        // Lance un ensemble de 500 itérations 
        u = solver.solveWithGuess(b_glob, u);
        it_total += solver.iterations();

        // Résidu "vrai" (extérieur au solveur)
        Eigen::VectorXd r = b_glob - Jop * u;   
        relres = r.norm() / (bnorm > 0 ? bnorm : 1.0);

        std::cout << "[adjoint] iters=" << it_total << "  relres=" << relres << std::endl;

        if (relres <= tol) break;

        // Divergence claire
        if (!std::isfinite(relres)) {
            throw std::runtime_error("BiCGSTAB diverged");
        }
    }

    if (relres > tol) {
        throw std::runtime_error("BiCGSTAB did not converge");
    }






    // Dépaquetage dans lambda
    lambda.assign(u.data(), u.data() + u.size());
}





/// 6) computeShapeGradient

void updateDerivedGeometry(Mesh& mesh) {

    for(std::size_t eid=0; eid<mesh.edges.size(); ++eid) {
        Edge& edge = mesh.edges[eid];
        // if (edge.boundaryCondition != 'w') continue; // il manque des conditions
        if ((edge.boundaryCondition != 'w') && 
            (mesh.vertices[edge.leftVertexID ].boundaryCondition != 'w') && 
            (mesh.vertices[edge.rightVertexID].boundaryCondition != 'w')) continue;

        
        // std::cout << "eid " << eid << " edge.edgeNormal.x " << edge.edgeNormal.x << " edge.edgeNormal.y " << edge.edgeNormal.y << std::endl;
        
        const auto leftVertexID  = edge.leftVertexID;
        const auto rightVertexID = edge.rightVertexID;
        const auto leftCellID    = edge.leftCellID;

        vector_3D leftVertex  = mesh.vertices[leftVertexID ].p;
        vector_3D rightVertex = mesh.vertices[rightVertexID].p;
        vector_3D edgeTangente = vector_3D{leftVertex.x - rightVertex.x, leftVertex.y - rightVertex.y, 0};
        edge.edgeLength = std::sqrt(edgeTangente.x * edgeTangente.x + edgeTangente.y * edgeTangente.y + edgeTangente.z * edgeTangente.z);
        edge.edgeNormal = vector_3D{edgeTangente.y / edge.edgeLength, - edgeTangente.x / edge.edgeLength, 0};

        Cell& cell = mesh.cells[leftCellID];
        // std::cout << "leftCellID " << leftCellID << " cell.cellVolume " << cell.cellVolume << " cell.centre.x " << cell.centre.x << " cell.centre.x " << cell.centre.y << std::endl;
        if (cell.verticeIDs.size() == 3) {
            vector_3D n0 = mesh.vertices[cell.verticeIDs[0]].p;
            vector_3D n1 = mesh.vertices[cell.verticeIDs[1]].p;
            vector_3D n2 = mesh.vertices[cell.verticeIDs[2]].p;
            vector_3D n0n1{n1.x - n0.x, n1.y - n0.y, 0};
            vector_3D n0n2{n2.x - n0.x, n2.y - n0.y, 0};
            cell.cellVolume = std::abs(0.5 * (n0n1.x * n0n2.y - n0n2.x * n0n1.y));
            cell.centre  = vector_3D{1.0/3.0 * (n0.x + n1.x + n2.x), 1.0/3.0 * (n0.y + n1.y + n2.y), 0};
        }
        else if (cell.verticeIDs.size() == 4) {
            vector_3D n0 = mesh.vertices[cell.verticeIDs[0]].p;
            vector_3D n1 = mesh.vertices[cell.verticeIDs[1]].p;
            vector_3D n2 = mesh.vertices[cell.verticeIDs[2]].p;
            vector_3D n3 = mesh.vertices[cell.verticeIDs[3]].p;
            vector_3D n0n1{n1.x - n0.x, n1.y - n0.y, 0};
            vector_3D n0n2{n2.x - n0.x, n2.y - n0.y, 0};
            vector_3D n2n1{n1.x - n2.x, n1.y - n2.y, 0};
            vector_3D n2n3{n3.x - n2.x, n3.y - n2.y, 0};
            cell.cellVolume = 0.5 * (std::abs(n0n1.x * n0n2.y - n0n2.x * n0n1.y) + std::abs(n2n1.x * n2n3.y - n2n3.x * n2n1.y));
            cell.centre = vector_3D{0.25 * (n0.x + n1.x + n2.x + n3.x), 0.25 * (n0.y + n1.y + n2.y + n3.y), 0};
        }
        // std::cout << "leftCellID " << leftCellID << " cell.cellVolume " << cell.cellVolume << " cell.centre.x " << cell.centre.x << " cell.centre.x " << cell.centre.y << std::endl;
        
        vector_3D centreLeftCell  = mesh.cells[leftCellID].centre;
        vector_3D centreEdge = vector_3D{0.5*(mesh.vertices[leftVertexID].p.x+mesh.vertices[rightVertexID].p.x), 0.5*(mesh.vertices[leftVertexID].p.y+mesh.vertices[rightVertexID].p.y), 0};
        vector_3D vectorCellLR{centreLeftCell.x - centreEdge.x, centreLeftCell.y - centreEdge.y, 0};
        if (vectorCellLR.x * edge.edgeNormal.x + vectorCellLR.y * edge.edgeNormal.y < 0)
            edge.edgeNormal = vector_3D{-edge.edgeNormal.x, -edge.edgeNormal.y, 0};
        
        edge.centre = centreEdge;

        // std::cout << "eid " << eid << " edge.edgeNormal.x " << edge.edgeNormal.x << " edge.edgeNormal.y " << edge.edgeNormal.y << std::endl;
    }
}

static inline double yc_naca4(double m, double p, double xr) {
    xr = std::clamp(xr, 0.0, 1.0);
    if (m==0.0) return 0.0;
    if (xr <= p) {
        return m/(p*p) * (2.0*p*xr - xr*xr);
    } else {
        return m/((1.0-p)*(1.0-p)) * ((1.0 - 2.0*p) + 2.0*p*xr - xr*xr);
    }
}

static inline double yt_naca4(double t, double c, double xr) {
    xr = std::clamp(xr, 0.0, 1.0);
    return 5.0 * t * c * (
        0.2969*std::sqrt(xr)
      - 0.1260*xr
      - 0.3516*xr*xr
      + 0.2843*xr*xr*xr
      - 0.1015*xr*xr*xr*xr
    );
}

void applyShapePerturbationNACA12(Mesh& mesh, const SimulationParams& params, const std::string& param_name, const std::vector<std::string>& param_names_all, const std::vector<double>& param_values_all, double eps) {

    const double pos_x = getParamOrDefault(param_names_all, param_values_all, "pos_x", 0.355);
    const double c     = getParamOrDefault(param_names_all, param_values_all, "c",     1.0  );
    const double m0    = getParamOrDefault(param_names_all, param_values_all, "m",     0.02 );
    const double p0    = getParamOrDefault(param_names_all, param_values_all, "p",     0.4  );
    const double t     = getParamOrDefault(param_names_all, param_values_all, "t",     0.12 );

    // std::cout   << " pos_x " << pos_x
    //             << " c " << c 
    //             << " m0 " << m0 
    //             << " p0 " << p0 
    //             << " t " << t 
    //             << "\n";

    // !!! zone figée : dernier dixième de la corde !!!
    const double freeze_start = 0.9;

    double m1 = m0;
    double p1 = p0;
    if (param_name == "cambrure" || param_name == "m") {
        m1 = m0 + eps;
    } else if (param_name == "position_cambrure" || param_name == "p") {
        p1 = p0 + eps;
    } else {
        // Paramètre non géré ici
        return;
    }

    for(std::size_t vid=0; vid<mesh.vertices.size(); ++vid) {
        Vertex& vertex = mesh.vertices[vid];
        if (vertex.boundaryCondition != 'w') continue;

        const double x_local = vertex.p.x - pos_x;
        const double xr = x_local / c;
        // std::cout << "x_local " << x_local << " xr " << xr << std::endl;

        // !!! Ne PAS déplacer les points du dernier dixième !!!
        if (xr >= freeze_start) continue;

        const double yc0 = yc_naca4(m0, p0, xr);
        const double yt0 = yt_naca4(t, c, xr);

        // std::cout << "m0 " << m0 << " p0 " << p0 << std::endl;
        // std::cout << "t " << t << " c " << c << std::endl;
        // std::cout << "yc0 " << yc0 << " yt0 " << yt0 << std::endl;

        const double s = (vertex.p.y >= yc0) ? +1.0 : -1.0; // extrados ou intrados

        const double yc1 = yc_naca4(m1, p1, xr);

        const double x_new = pos_x + xr*c;
        const double y_new = yc1 + s * yt0;
        // std::cout << "yc1 " << yc1 << " yt0 " << yt0 << std::endl;

        // std::cout << "x_old " << vertex.p.x << " y_old " << vertex.p.y << std::endl;
        vertex.p.x = x_new;
        vertex.p.y = y_new;
        // std::cout << "x_new " << vertex.p.x << " y_new " << vertex.p.y << std::endl;
    }

    updateDerivedGeometry(mesh); 
}

// !!! pas de terme visqueux !!!
double computeDirectObjectiveTerm(Mesh& mesh, const SimulationParams& params, double beta=1) { // poids de la trainée
    double gamma = params.gamma;
    double rho_inf = params.rho_ref;
    double p_inf   = params.p_ref; 
    const double a_inf   = sound_speed_ideal(gamma,p_inf,rho_inf);
    double M_inf  = params.inflow_velocity;   
    double U_inf = M_inf * a_inf;

    double q_inf = 0.5 * (rho_inf * U_inf*U_inf);
    double corde = 1; // !!! Attention : à adapter

    double C_L = 0.0, C_D = 0.0;

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
        // // !!! pas de terme visqueux !!!
        // double tau_xx = 0,
        //        tau_xy_yx = 0,
        //        tau_yy = 0;
        
        L += (-p * n.y + tau_xy_yx * n.x + tau_yy * n.y) * A;
        D += (-p * n.x + tau_xx * n.x + tau_xy_yx * n.y) * A;
    }

    C_L = L / (q_inf * corde);
    C_D = D / (q_inf * corde);
    
    return - C_L + beta * C_D;
}


void computeShapeGradient(Mesh& mesh,
                          const SimulationParams& params,
                          const std::vector<double>& lambda,
                          const std::vector<std::string>& param_names_all,
                          const std::vector<double>& param_values_all,
                          const std::vector<std::string>& param_names,
                          std::vector<double>& grad_out,
                          double beta=1,
                          double eps = 1e-6) {

    const size_t N_cells = mesh.cells.size();
    const size_t N_glob  = 4 * N_cells;

    std::vector<double> R0_rho(N_cells), R0_rho_u(N_cells), R0_rho_v(N_cells), R0_E(N_cells);
    computeResidual(mesh, params, R0_rho, R0_rho_u, R0_rho_v, R0_E); 
    
    
    // Aplatisseur de R sans R0
    auto pack_diff = [&](const std::vector<double>& r,
                         const std::vector<double>& ru,
                         const std::vector<double>& rv,
                         const std::vector<double>& E,
                         std::vector<double>& out) {
        out.resize(N_glob);
        // la soustraction n'est pas vraiment utile !!!
        for(size_t i=0;i<N_cells;++i){
            out[4*i+0] = r[i]; //  - R0_rho[i]; 
            out[4*i+1] = ru[i]; // - R0_rho_u[i];
            out[4*i+2] = rv[i]; // - R0_rho_v[i];
            out[4*i+3] = E[i]; //  - R0_E[i];
        }
    };

    // Produit scalaire lambda^T v
    auto dot_lambda = [&](const std::vector<double>& v)->double {
        double s = 0.0;
        for(size_t k=0;k<N_glob;++k) s += lambda[k]*v[k];
        return s;
    };

    grad_out.assign(param_names.size(), 0.0);


    // différence centrée +eps et -eps
    for(size_t k=0; k<param_names.size(); ++k){
        const std::string& param_name = param_names[k];

        Mesh mesh_plus = mesh;
        applyShapePerturbationNACA12(mesh_plus, params, param_name, param_names_all, param_values_all, eps);
        std::vector<double> Rp_rho(N_cells), Rp_rho_u(N_cells), Rp_rho_v(N_cells), Rp_E(N_cells);
        computeResidual(mesh_plus, params, Rp_rho, Rp_rho_u, Rp_rho_v, Rp_E); 

        Mesh mesh_minus = mesh;
        applyShapePerturbationNACA12(mesh_minus, params, param_name, param_names_all, param_values_all, -eps);
        std::vector<double> Rm_rho(N_cells), Rm_rho_u(N_cells), Rm_rho_v(N_cells), Rm_E(N_cells);
        computeResidual(mesh_minus, params, Rm_rho, Rm_rho_u, Rm_rho_v, Rm_E); 


        std::vector<double> dR_glob(N_glob), Rplus_R0(N_glob), Rminus_R0(N_glob);
        pack_diff(Rp_rho, Rp_rho_u, Rp_rho_v, Rp_E, Rplus_R0); 
        pack_diff(Rm_rho, Rm_rho_u, Rm_rho_v, Rm_E, Rminus_R0);

        dR_glob.resize(N_glob);
        for(size_t i=0;i<N_glob;++i)
            dR_glob[i] = (Rplus_R0[i] - Rminus_R0[i]) / (2.0*eps);


        double J_plus  = computeDirectObjectiveTerm(mesh_plus,  params, beta);
        double J_minus = computeDirectObjectiveTerm(mesh_minus, params, beta);
        double dJ_direct = (J_plus - J_minus) / (2.0*eps);

        // dJ/dθ_k = (partial J/partial θ_k) - lambda^T (partial R/partial θ_k) 
        grad_out[k] = dJ_direct - dot_lambda(dR_glob);
    }
}




/// 7) optimizeShape
void optimizeShape(Mesh& mesh, const SimulationParams& params, double eps) {

    std::vector<double> param_values_all;
    std::vector<std::string> param_names, param_names_all;

    loadParamsGeneric(params.shape_param_file, param_names_all, param_values_all, param_names);

    // std::cout << "[debug] param_names_all.size=" << param_names_all.size() << " param_values_all.size=" << param_values_all.size() << "\n";
    // std::cout << "[debug] param_names (optimisés) = ";
    // for (auto& s : param_names) std::cout << s << " ";
    // std::cout << "\n";

    double beta = 1;

    const double J_current = computeDirectObjectiveTerm(mesh, params, beta);

    std::vector<double> lambda;
    solveAdjointSystem(mesh, params, lambda, beta, eps);
    
    std::vector<double> grad; // taille = param_names.size()
    computeShapeGradient(mesh, params, lambda, param_names_all, param_values_all, param_names, grad, beta, eps);


    // Affichage grad
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "[preview] J courant = " << J_current << "\n";
    std::cout << "[preview] Gradient de forme (dJ/dθ) et sens pour diminuer J :\n";

    for (std::size_t k = 0; k < param_names.size(); ++k) {
        const double g = grad[k];
        const char* sens =
            (g > 0.0) ? "-> diminuer ce paramètre (pas négatif)"
          : (g < 0.0) ? "-> augmenter ce paramètre (pas positif)"
                      : "-> gradient ~ 0 (aucune direction claire)";

        std::cout << "  - " << param_names[k]
                  << " : dJ/d" << param_names[k] << " = " << g
                  << "    " << sens << "\n";
    }

    appendShapeOptLog(params, param_names, J_current, grad);
    
}






} // namespace navier_stokes
















