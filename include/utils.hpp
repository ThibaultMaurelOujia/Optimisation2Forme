#ifndef NAVIER_STOKES_CPP_UTILS_HPP
#define NAVIER_STOKES_CPP_UTILS_HPP

#include <filesystem>
#include <cmath> 
#include <cassert>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

#include <omp.h>
#include <gmsh.h>

#include "mesh.hpp"
#include "params.hpp"

namespace navier_stokes {


inline constexpr double PI     = 3.14159265358979323846;
inline constexpr double TWO_PI = 2.0 * PI;


using Clock      = std::chrono::high_resolution_clock;
using TimePoint  = Clock::time_point;


// DEmarre le chronometre
TimePoint time_tic();

// ArrEte le chrono, affiche (si msg non vide) et renvoie le temps ecoule en secondes
double time_toc(const TimePoint& t0, const std::string& msg = "", int verbosity=1);


struct Mesh;
struct SimulationParams;


void loadCellStates(Mesh& mesh, const SimulationParams& params);

void saveCellStates(const Mesh& mesh, const SimulationParams& params);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_UTILS_HPP