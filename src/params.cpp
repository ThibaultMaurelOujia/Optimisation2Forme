#include "params.hpp"

namespace navier_stokes {

void SimulationParams::finalize() {
    cv = cp / gamma; 
    R  = cp - cv;   
}

// trim whitespace from both ends of a string
static inline std::string trim(const std::string &s) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return ""; 
    auto last  = s.find_last_not_of (" \t\r\n");
    return s.substr(first, last - first + 1);
}

bool load_config(const std::string &filename, SimulationParams &p) {
    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    while (std::getline(in, line)) {
        // ignore comments and empty lines
        auto t = trim(line);
        if (t.empty() || t[0] == '#') continue;

        auto eq = t.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(t.substr(0, eq));
        std::string val = trim(t.substr(eq + 1));
        std::istringstream ss(val);

        if      (key == "mesh_name")            p.mesh_name = val;
        else if (key == "Lx")                   ss >> p.Lx;
        else if (key == "Ly")                   ss >> p.Ly;
        else if (key == "Nx")                   ss >> p.Nx;
        else if (key == "Ny")                   ss >> p.Ny;

        else if (key == "cfl_number")           ss >> p.cfl_number;
        else if (key == "final_time")           ss >> p.final_time;
        
        else if (key == "gamma")                ss >> p.gamma;
        else if (key == "viscosity")            ss >> p.viscosity;
        else if (key == "Pr")                   ss >> p.Pr;
        else if (key == "cp")                   ss >> p.cp;
        else if (key == "cv")                   ss >> p.cv;
        else if (key == "R")                    ss >> p.R;

        else if (key == "advection_scheme")     p.advection_scheme    = val;
        else if (key == "boundary_conditions")  p.boundary_conditions = val;

        else if (key == "initial_condition")    p.initial_condition   = val;
        else if (key == "noise_level")          ss >> p.noise_level;

        else if (key == "obstacle_type")        p.obstacle_type       = val;
        else if (key == "obstacle_center_x")    ss >> p.obstacle_center_x;
        else if (key == "obstacle_center_y")    ss >> p.obstacle_center_y;
        else if (key == "obstacle_size")        ss >> p.obstacle_size;

        else if (key == "inflow_velocity")      ss >> p.inflow_velocity;
        else if (key == "rho_ref")              ss >> p.rho_ref;
        else if (key == "p_ref")                ss >> p.p_ref;

        else if (key == "output_directory")     p.output_directory    = val;
        else if (key == "save_interval")        ss >> p.save_interval;
        else if (key == "state_save_path")      p.state_save_path     = val;
        else if (key == "state_load_path")      p.state_load_path     = val;
        else if (key == "output_quantity")      p.output_quantity     = val;
        else if (key == "verbosity")            ss >> p.verbosity;
        else if (key == "reload_config")        ss >> p.reload_config;

        else if (key == "shape_opt_enable")     ss >> p.shape_opt_enable;
        // else if (key == "shape_opt_max_iters")  ss >> p.shape_opt_max_iters;
        else if (key == "shape_param_file")     p.shape_param_file    = val;
        else if (key == "shape_log_file")       p.shape_log_file       = val;
 
    }
 
    p.finalize();
    return true;
}

void print_params(const SimulationParams& p) {
    std::cout
    << "=== Simulation parameters ===\n"
    << " Domain:      Lx=" << p.Lx << " ; Ly=" << p.Ly << "\n"
    << " Mesh:        Nx=" << p.Nx << " ; Ny=" << p.Ny << "\n"

    << "=== Physical properties: ===\n"
    << " Viscosity:     " << p.viscosity << "\n"
    << " Pr:            " << p.Pr<< "\n\n"

    << "=== Inflow conditions: ===\n"
    << " Mach:          " << p.inflow_velocity << "\n"
    << " rho_ref:       " << p.rho_ref   << " kg/m^3\n"
    << " p_ref:         " << p.p_ref     << " Pa\n\n"

    << "=== Numerical settings: ===\n"
    << " CFL:           " << p.cfl_number << "\n"
    << " Advection:     " << p.advection_scheme << "\n"
    << " BCs:           " << p.boundary_conditions << "\n"
    << " Init:          " << p.initial_condition << "\n"
    << " Obstacle:      " << p.obstacle_type << "\n\n"

    << "=== Output & logging: ===\n"
    << " Out dir:       " << p.output_directory << "\n"
    << " Save every:    " << p.save_interval << "\n"
    << " Verbosity:     " << p.verbosity << "\n"
    << " Final time:    " << p.final_time << "\n"
    << "=============================\n"
    << std::flush;
  }
  
} // namespace navier_stokes


