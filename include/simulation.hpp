#ifndef NAVIER_STOKES_CPP_SIMULATION_HPP
#define NAVIER_STOKES_CPP_SIMULATION_HPP

#include "utils.hpp"
#include "params.hpp"
#include "init.hpp"
#include "boundary.hpp"
#include "advection.hpp"
#include "diffusion.hpp"
#include "compute.hpp"
#include "shape_optimization.hpp"

#include "renderer_colormap.hpp"
#include <SFML/Graphics.hpp>

#include <vector>


namespace navier_stokes {


class Simulator {
public:
    explicit Simulator(Mesh& mesh, const SimulationParams& params);

    void run();

private:
    Mesh& mesh_;
    SimulationParams params_;

    std::pair<double,double> display_size_;

    void initialize_fields();  
    void time_loop();  
    void output_step(int iter, double t, double dt);  
    
    void step_SSP_RK3(double dt);
    void rk3_substep(double c0, double c1, double dt);

     // SFML objects for on-screen rendering 
    sf::RenderWindow               win_;       // fenetre unique
    sf::Image                      img_;
    sf::Texture                    tex_;
    sf::Sprite                     spr_;
    
};



} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_SIMULATION_HPP