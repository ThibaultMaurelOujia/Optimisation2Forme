#include "simulation.hpp"

namespace navier_stokes {




Simulator::Simulator(Mesh& mesh, const SimulationParams& params): mesh_(mesh), params_{params}, 
                                                    display_size_{ compute_display_size(params.Lx, params.Ly) }, 
                                                    win_{
                                                        sf::VideoMode{
                                                            sf::Vector2u{
                                                                unsigned(display_size_.first  * 100),
                                                                unsigned(display_size_.second * 100)
                                                            }
                                                        },
                                                        "Navier-Stokes"
                                                    }, 
                                                    img_{ sf::Vector2u{
                                                            static_cast<unsigned>(params.Nx),
                                                            static_cast<unsigned>(params.Ny)
                                                        },
                                                        sf::Color::Black
                                                        }, 
                                                    tex_{ img_ }, 
                                                    spr_{ tex_ }
                                                    {

    // fenetre SFML 
    // mise a l'echelle du sprite 
    spr_.setScale(
        sf::Vector2f{
            float(win_.getSize().x) / float(params.Nx),
            float(win_.getSize().y) / float(params.Ny)
        }
    );
    win_.setVerticalSyncEnabled(true);
}

void Simulator::run() {
    initialize_fields();        // fill u,v,p,mask at t=0 
    time_loop();
}

void Simulator::initialize_fields() {
    // 1) initial condition
    initialize_flow_field(mesh_, params_);



    // const std::vector<double> param_values; // a changer plus tard !!!
    // applyShapePerturbation(mesh_, params_, "m", param_values, 1e-6);
    // std::cerr << "STOP\n";
    // std::exit(EXIT_FAILURE);



    // debugVisualizeScalarsUnstructured(mesh_, img_, tex_, spr_, win_); 
    // debugVisualizeVertexContributions(mesh_, img_, tex_, spr_, win_); 

    // std::cerr << "STOP\n";
    // std::exit(EXIT_FAILURE);

}

void Simulator::time_loop() {
    double t = 0.0;
    int    it = 0;

    loadCellStates(mesh_, params_);

    const auto& cells = mesh_.cells;
    const auto& edges = mesh_.edges;

    while (t < params_.final_time) {
        // 1) calcul des vitesses caractéristiques max et du plus petit rayon de cellule
        double max_wave = 1e-8;
        double min_h    = std::numeric_limits<double>::infinity();

        #pragma omp parallel for reduction(max:max_wave) reduction(min:min_h)
        for (std::size_t cid = 0; cid < cells.size(); ++cid) {
            const Cell& cell = cells[cid];
 
            double rho = cell.rho;
            double ru  = cell.rho_u;
            double rv  = cell.rho_v;
            double E   = cell.E;

            double u = ru / rho;
            double v = rv / rho;
            double kinetic = 0.5 * (ru*ru + rv*rv) / rho;
            double p = (params_.gamma - 1.0) * (E - kinetic);
            double c = std::sqrt(params_.gamma * p / rho);

            double speed = std::hypot(u, v) + c; 
            max_wave = std::max(max_wave, speed);

            // rayon de la cellule, CFL non structuré
            double perimeter = 0.0;
            for (auto eid : cell.edgeIDs) {
                perimeter += edges[eid].edgeLength;
            } 
            double h = 2.0 * cell.cellVolume / perimeter;
            min_h = std::min(min_h, h);
        }

        // dt advectif et diffusif
        double dt_adv  = params_.cfl_number * min_h / max_wave;
        double dt_diff = 0.5 * (min_h * min_h) / params_.viscosity;
        double dt      = std::min(dt_adv, dt_diff);
 

        step_SSP_RK3(dt);
 
        if (it % params_.save_interval == 0)
            output_step(it, t, dt);

        if (it % params_.reload_config == 0 && params_.reload_config != 0)
            load_config("config.txt", params_);
 
        t  += dt;
        ++it;
    }

    if (params_.shape_opt_enable)
        saveCellStates(mesh_, params_);


    // std::cout << "[Test] Lancement de optimizeShape" << std::endl;
    optimizeShape(mesh_,  params_, 1e-6);
}



void Simulator::step_SSP_RK3(double dt) {
    #pragma omp parallel for 
    for(std::size_t cid=0; cid<mesh_.cells.size(); ++cid) {
        Cell& cell = mesh_.cells[cid]; 
        cell.rho_old = cell.rho; cell.rho_u_old = cell.rho_u; cell.rho_v_old = cell.rho_v; cell.E_old = cell.E;
    }
 
    rk3_substep(0.0, 1.0, dt);
 
    rk3_substep(3.0/4.0, 1.0/4.0, dt);
 
    rk3_substep(1.0/3.0, 2.0/3.0, dt);
}


void Simulator::rk3_substep(double c0, double c1, double dt) {

    // 1) advection term
    auto t_adv = time_tic();
    computeAdvectionTerm(mesh_, params_);
    time_toc(t_adv, "Advection", params_.verbosity); 

    // 2) diffusion term
    auto t_diff = time_tic();
    computeDiffusionTerm(mesh_, params_);
    time_toc(t_diff, "Diffusion", params_.verbosity); 

    // 3) RK3 linear combination
    auto t_sim = time_tic();
    #pragma omp parallel for 
    for(std::size_t cid=0; cid<mesh_.cells.size(); ++cid){
        Cell& cell = mesh_.cells[cid];

        cell.rho   = c0 * cell.rho_old   + c1 * (cell.rho   + dt * (cell.R_rho)); 
        cell.rho_u = c0 * cell.rho_u_old + c1 * (cell.rho_u + dt * (cell.R_rho_u)); 
        cell.rho_v = c0 * cell.rho_v_old + c1 * (cell.rho_v + dt * (cell.R_rho_v)); 
        cell.E     = c0 * cell.E_old     + c1 * (cell.E     + dt * (cell.R_E)); 
    }
    time_toc(t_sim, "RK3 linear combination", params_.verbosity); 
}




void Simulator::output_step(int it, double t, double dt) {
    if (params_.verbosity > 0)
        std::cout << "it=" << it  << "  t=" << t  << "  dt=" << dt << '\n' << std::flush; // std::endl


    double C_L, C_D;
    computeLiftDragCoefficients(mesh_, params_, C_L, C_D);
    std::cout << "C_L=" << C_L  << " ; C_D=" << C_D << '\n' << std::flush;


    computeCellScalar(mesh_, params_.output_quantity, params_);
    double scalar_min = mesh_.scalar_min, scalar_max = mesh_.scalar_max;
    if (params_.output_quantity == "p") {
        // scalar_min = 75000; scalar_max = 125000;
        const double step_min_max = 5000.0;
        scalar_min = std::floor(mesh_.scalar_min/step_min_max)*step_min_max;
        scalar_max = std::ceil (mesh_.scalar_max/step_min_max)*step_min_max;
        // scalar_min = 0; scalar_max = 1e6;
        render_scalar_field_unstructured(mesh_, img_, scalar_min, scalar_max, "");
    }
    render_scalar_field_unstructured(mesh_, img_, scalar_min, scalar_max, "");
    // render_scalar_field_unstructured(mesh_, img_, mesh_.scalar_min, mesh_.scalar_max, "");
    std::cout << "mesh_.scalar_min " << mesh_.scalar_min << " mesh_.scalar_max " << mesh_.scalar_max << '\n' << std::flush;

    // on met a jour la texture, on reassocie au sprite
    tex_.update(img_);
    spr_.setTexture(tex_, true);

    // on dessine
    win_.clear();
    win_.draw(spr_);
    win_.display();

    // boucle d'evenements SFML3
    while (auto ev = win_.pollEvent()) {
        if (ev->is<sf::Event::Closed>())
            win_.close();
    }


    // Noms de fichiers
    const std::string base = params_.output_directory + "/it_" + std::to_string(it);
    const std::string file_u = base + "_u.bin";
    const std::string file_v = base + "_v.bin";
    const std::string file_w = base + "_w.bin";

    // Écriture binaire de u_ et v_
    try {
        // write_field_binary(file_u, u_, Nx, Ny, N_ghost);
        // write_field_binary(file_v, v_, Nx, Ny, N_ghost);
        // write_field_binary(file_w, w, Nx, Ny, N_ghost);
    }
    catch (const std::exception& e) {
        std::cerr << "Erreur ecriture binaire : " << e.what() << "\n";
    }
}

} // namespace navier_stokes


















