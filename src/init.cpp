#include "init.hpp"

namespace navier_stokes {




inline double sound_speed_ideal(double gamma, double p, double rho){ return std::sqrt(gamma * p / rho); }

inline void primitives_to_conserved(double rho, double u, double v, double p, double gamma,
                                    double& rho_u, double& rho_v, double& E){
    rho_u = rho * u;
    rho_v = rho * v;
    double kinetic = 0.5 * rho * (u*u + v*v);
    E = p/(gamma-1.0) + kinetic;
}


//---------------------------------------------------------------------------
void init_kelvin_helmholtz(Mesh& mesh, const SimulationParams& params,   
                           double delta = 0.005,  
                           double amp   = 1e-1, 
                           int    kx    = 4)  
{
    const double Lx = params.Lx, Ly = params.Ly;
    const double gamma = params.gamma;

    double rho0 = params.rho_ref;  
    double p0   = params.p_ref;    
    const double a0   = sound_speed_ideal(gamma,p0,rho0);
 
    double M  = params.inflow_velocity;   
    double U0 = M * a0;   
    
    const double two_delta2 = 2.0 * delta * delta;

    for(std::size_t i = 0; i<mesh.cells.size(); ++i) {
        Cell& cell = mesh.cells[i];
        double x = cell.centre.x;
        double y = cell.centre.y;

        double u   = U0 * std::tanh((y-0.5*Ly)/delta);      
        double v   = amp * std::sin(2*M_PI*kx*x/Lx)
                            * std::exp(-(y-0.5*Ly)*(y-0.5*Ly)/two_delta2);

        double rho_u_, rho_v_, E_;
        primitives_to_conserved(rho0,u,v,p0,gamma,rho_u_,rho_v_,E_);
        
        cell.rho   = rho0;
        cell.rho_u = rho_u_;
        cell.rho_v = rho_v_;
        cell.E     = E_;
    }
}



//---------------------------------------------------------------------------
void init_one_x(Mesh& mesh, const SimulationParams& params) 
{
    const double gamma=params.gamma;
 
    double rho0 = params.rho_ref;
    double p0   = params.p_ref;
    const double a0   = sound_speed_ideal(gamma,p0,rho0);

    double M  = params.inflow_velocity;    
    double u0 = M * a0;

    for(std::size_t i = 0; i<mesh.cells.size(); ++i) {
        Cell& cell = mesh.cells[i];
        
        double rho_u_,rho_v_,E_;
        primitives_to_conserved(rho0,u0,0.0,p0,gamma,rho_u_,rho_v_,E_);
        
        cell.rho   = rho0;
        cell.rho_u = rho_u_;
        cell.rho_v = rho_v_;      // = 0
        cell.E     = E_;
    }
}

void init_one_x_gaussian_vortex(Mesh& mesh,
                                const SimulationParams& params,
                                double A = 0.10)      
{
    const double gamma = params.gamma;
    const double rho0  = params.rho_ref;
    const double p0    = params.p_ref;
    const double a0    = std::sqrt(gamma * p0 / rho0);

    const double M  = params.inflow_velocity;     
    const double u0 = M * a0;             
 
    const double r0 = 0.25 * std::min(params.Lx, params.Ly);
    const double r0_inv2 = 1.0 / (r0 * r0);
 
    const double xc = 0.5 * (mesh.x_min + mesh.x_max);
    const double yc = 0.5 * (mesh.y_min + mesh.y_max);

    for (Cell& cell : mesh.cells) {
 
        double dx = cell.centre.x - xc;
        double dy = cell.centre.y - yc;
        double r2 = dx*dx + dy*dy;
        double g  = std::exp(-r2 * r0_inv2);    
 
        double du = - A * dy * g;    
        double dv =   A * dx * g;   
 
        double u = u0 + du;
        double v =      dv;

        double ru, rv, Et;
        primitives_to_conserved(rho0, u, v, p0, gamma, ru, rv, Et);

        cell.rho   = rho0;
        cell.rho_u = ru;
        cell.rho_v = rv;
        cell.E     = Et;
    }
}

void init_one_x_gaussian_density(Mesh& mesh,
                                 const SimulationParams& params,
                                 double eps = 0.10)    
{
    const double gamma = params.gamma;
    const double rho0  = params.rho_ref;
    const double p0    = params.p_ref;
    const double a0    = std::sqrt(gamma * p0 / rho0);

    const double M  = params.inflow_velocity;
    const double u0 = M * a0;   

    const double r0 = 0.25 * std::min(params.Lx, params.Ly);
    const double r0_inv2 = 1.0 / (r0 * r0);

    const double xc = 0.5 * (mesh.x_min + mesh.x_max);
    const double yc = 0.5 * (mesh.y_min + mesh.y_max);

    for (Cell& cell : mesh.cells) {

        double dx = cell.centre.x - xc;
        double dy = cell.centre.y - yc;
        double r2 = dx*dx + dy*dy;
        double g  = std::exp(-r2 * r0_inv2);   

        double rho = rho0 * (1.0 + eps * g);  
        double u   = u0;
        double v   = 0.0;

        double ru, rv, Et;
        primitives_to_conserved(rho, u, v, p0, gamma, ru, rv, Et);

        cell.rho   = rho;
        cell.rho_u = ru;
        cell.rho_v = rv;
        cell.E     = Et;
    }
}

void init_one_x_gaussian_temperature(Mesh& mesh,
                                     const SimulationParams& params,
                                     double A = 0.10) {  
    const double gamma = params.gamma;
    const double rho0  = params.rho_ref;
    const double p0    = params.p_ref;
    const double E0    = p0/(gamma-1.0);

    const double r0       = 0.25 * std::min(params.Lx, params.Ly);
    const double r0_inv2  = 1.0/(r0*r0);
    const double xc = 0.5*(mesh.x_min + mesh.x_max);
    const double yc = 0.5*(mesh.y_min + mesh.y_max);

    for (Cell& c : mesh.cells) {
        double dx = c.centre.x - xc;
        double dy = c.centre.y - yc;
        double g  = std::exp(- (dx*dx + dy*dy)*r0_inv2 );

        double E = E0 * (1.0 + A * g);

        c.rho   = rho0;
        c.rho_u = 0.0;
        c.rho_v = 0.0;
        c.E     = E;
    }
}


//---------------------------------------------------------------------------
// void init_sod_x(Mesh& mesh, const SimulationParams& params) 
// {
//     const int Nx=params.Nx, Ny=params.Ny, N=params.bc_ghost_layers;
//     double gamma=params.gamma;

//     const double rho_L=1.0,  p_L=1.0;
//     const double rho_R=0.125,p_R=0.1;
//     const double a_L=sound_speed_ideal(gamma,p_L,rho_L);
//     const double a_R=sound_speed_ideal(gamma,p_R,rho_R);

//     for(std::size_t i = 0; i<mesh.cells.size(); ++i) {
//         Cell& cell = mesh.cells[i];
//         double x = cell.centre.x;
//         bool left = x - mesh.x_min < params.Lx/2;
//         double rho0 = left?rho_L:rho_R;
//         double p0   = left?p_L  :p_R;
        
//         double rho_u_,rho_v_,E_;
//         primitives_to_conserved(rho0,0.0,0.0,p0,gamma,rho_u_,rho_v_,E_);

//         cell.rho   = rho0;
//         cell.rho_u = rho_u_;
//         cell.rho_v = rho_v_;
//         cell.E     = E_;
//     }
// }
void init_sod_x(Mesh& mesh, const SimulationParams& params)
{
    const double gamma  = params.gamma;
    const double rho_ref = params.rho_ref; 
    const double p_ref   = params.p_ref;  

    const double rho_L = rho_ref * 1.0;
    const double p_L   = p_ref   * 1.0;
    const double rho_R = rho_ref * 0.125;
    const double p_R   = p_ref   * 0.1;
 
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        Cell& cell = mesh.cells[i];
        
        bool left = (cell.centre.x - mesh.x_min) < 0.5 * params.Lx;

        double rho0 = left ? rho_L : rho_R;
        double p0   = left ? p_L   : p_R;
        double u0   = 0.0;
        double v0   = 0.0;

        double rho_u_, rho_v_, E_;
        primitives_to_conserved(rho0, u0, v0, p0, gamma, rho_u_, rho_v_, E_);

        cell.rho   = rho0;
        cell.rho_u = rho_u_;
        cell.rho_v = rho_v_;
        cell.E     = E_;
    }
}


//---------------------------------------------------------------------------
void init_isentropic_vortex(Mesh& mesh, const SimulationParams& params)
{
    const double Lx    = params.Lx;
    const double Ly    = params.Ly;
    const double gamma = params.gamma;

    const double U0   = 1.0;
    const double V0   = 0.0;
    const double beta = 5.0;
    const double x0   = 0.5 * Lx;
    const double y0   = 0.5 * Ly;

    for (Cell& cell : mesh.cells) {
        const double x = cell.centre.x;
        const double y = cell.centre.y;

        const double xd   = x - x0;
        const double yd   = y - y0;
        const double r2   = xd * xd + yd * yd;
        const double expf = std::exp((1.0 - r2) / 2.0);

        const double u = U0 - beta / (2.0 * M_PI) * yd * expf;
        const double v = V0 + beta / (2.0 * M_PI) * xd * expf;

        const double T     = 1.0
            - (gamma - 1.0) / (2.0 * gamma)
              * (beta * beta / (4.0 * M_PI * M_PI)) * expf * expf;
        const double rho0  = std::pow(T, 1.0 / (gamma - 1.0));
        const double p0    = std::pow(T,  gamma / (gamma - 1.0));

        double ru, rv, Et;
        primitives_to_conserved(rho0, u, v, p0, gamma, ru, rv, Et);

        cell.rho   = rho0;
        cell.rho_u = ru;
        cell.rho_v = rv;
        cell.E     = Et;
    }
}



//---------------------------------------------------------------------------
void init_blast_2d(Mesh& mesh, const SimulationParams& params)
{
    const double Lx    = params.Lx;
    const double Ly    = params.Ly;
    const double gamma = params.gamma;

    constexpr double rho_in  = 1.0,  p_in  = 1000.0;
    constexpr double rho_out = 1.0,  p_out = 0.1;
    const double x0 = 0.5 * Lx;
    const double y0 = 0.5 * Ly;
    const double r0 = 0.1 * std::min(Lx, Ly);

    for (Cell& cell : mesh.cells) {
        const double x = cell.centre.x;
        const double y = cell.centre.y;
        const double r = std::hypot(x - x0, y - y0);

        const double rho0 = (r < r0 ? rho_in : rho_out);
        const double p0   = (r < r0 ? p_in  : p_out);

        double ru, rv, Et;
        primitives_to_conserved(rho0, 0.0, 0.0, p0, gamma, ru, rv, Et);

        cell.rho   = rho0;
        cell.rho_u = ru;
        cell.rho_v = rv;
        cell.E     = Et;
    }
}



//---------------------------------------------------------------------------
void init_sin_cos_velocity(Mesh& mesh, const SimulationParams& params) {
    const double gamma = params.gamma;

    const double rho0 = params.rho_ref;  
    const double p0   = params.p_ref;   

    for (Cell& cell : mesh.cells) {
        const double x = cell.centre.x / params.Lx * TWO_PI;// + 2;// + 1;
        const double y = cell.centre.y / params.Ly * TWO_PI;// + 1;// + 1;
        
        const double u = std::sin(x) * std::cos(y);
        const double v = 0.0;
        
        double rho_u, rho_v, E;
        primitives_to_conserved(rho0, u, v, p0, gamma, rho_u, rho_v, E);

        cell.rho   = rho0;
        cell.rho_u = rho_u;
        cell.rho_v = rho_v;   // = 0
        cell.E     = E;
    }
}

void computeDivergence_sin_cos(Mesh& mesh, const SimulationParams& params) {
    const double kx = TWO_PI / params.Lx;
    const double ky = TWO_PI / params.Ly;
    for (Cell& cell : mesh.cells) {
        double X = cell.centre.x;
        double Y = cell.centre.y;
        cell.scalar = kx * std::cos(kx * X) * std::cos(ky * Y);
    }
}


//---------------------------------------------------------------------------
static void add_velocity_noise(Mesh& mesh, double amplitude)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> d(-amplitude,amplitude);

    for(Cell& cell : mesh.cells) {
        cell.rho_u += d(gen);
        cell.rho_v += d(gen);
    }
}



//---------------------------------------------------------------------------
void initialize_flow_field(Mesh& mesh, const SimulationParams& params) {
    if      (params.initial_condition == "kelvin_helmholtz")  init_kelvin_helmholtz(mesh, params);
    else if (params.initial_condition == "one_x")             init_one_x(mesh, params);
    else if (params.initial_condition == "one_x_vortex")      init_one_x_gaussian_vortex(mesh, params);
    else if (params.initial_condition == "one_x_density")     init_one_x_gaussian_density(mesh, params);
    else if (params.initial_condition == "one_x_temp")        init_one_x_gaussian_temperature(mesh, params);
    else if (params.initial_condition == "sod_x")             init_sod_x(mesh, params);
    else if (params.initial_condition == "isentropic_vortex") init_isentropic_vortex(mesh, params);
    else if (params.initial_condition == "blast_2d")          init_blast_2d(mesh, params);
    else if (params.initial_condition == "sin_cos")           init_sin_cos_velocity(mesh, params);
    else
        throw std::invalid_argument("Unknown initial condition: " + params.initial_condition);

    // bruit sur les vitesses conservées
    add_velocity_noise(mesh, params.noise_level);
}

// isentropic_vortex, blast_2d, riemann_2d



} // namespace navier_stokes








