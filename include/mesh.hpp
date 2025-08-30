#ifndef NAVIER_STOKES_CPP_MESH_HPP
#define NAVIER_STOKES_CPP_MESH_HPP

#include "utils.hpp"
#include "params.hpp"

#include <gmsh.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace navier_stokes {


using Key   = std::pair<std::size_t,std::size_t>;
using Value = std::pair<std::size_t,std::size_t>;


struct vector_3D {
  double x, y, z;
};

struct Vertex {
    vector_3D p;

    std::unordered_set<std::size_t> edgeIDs;
    std::unordered_set<std::size_t> cellIDs;
    std::vector<std::size_t> cellIDsOrdered;

    char boundaryCondition = '\0';
    
    // double rho;
    // double rho_u;
    // double rho_v;
    // double E;
};

struct Edge {
    vector_3D centre;
    
    std::size_t leftVertexID, rightVertexID;
    std::size_t leftCellID = - 1, rightCellID = - 1; 

    char boundaryCondition = '\0';

    double edgeLength;
    
    vector_3D edgeNormal; // normale unitaire pointant vers leftCell
    
    double rho_L;
    double rho_u_L;
    double rho_v_L;
    double E_L;
    double rho_R;
    double rho_u_R;
    double rho_v_R;
    double E_R;
    
    double tau_xx = 0.0;
    double tau_xy_yx = 0.0;           // tau_xy and tau_yx
    double tau_yy = 0.0;
};

struct Cell {
    vector_3D centre;

    std::vector<std::size_t> verticeIDs;
    std::unordered_set<std::size_t> edgeIDs; 
    std::unordered_set<std::size_t> neighbourIDs; 

    bool border = false;

    double cellVolume;

    double rho;
    double rho_u;
    double rho_v;
    double E;
    
    double rho_x;
    double rho_u_x;
    double rho_v_x;
    double E_x;
    double rho_y;
    double rho_u_y;
    double rho_v_y;
    double E_y;

    double R_rho;
    double R_rho_u;
    double R_rho_v;
    double R_E;

    double rho_old;
    double rho_u_old;
    double rho_v_old;
    double E_old;

    double scalar;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Cell> cells;

    std::unordered_map<std::size_t,std::size_t> vertexMap; 

    double scalar_min, scalar_max;

    double x_min, x_max, y_min, y_max, Lx, Ly; 

    std::vector<std::array<std::size_t,3>> pixelsToCellIDs;

    std::pair<double,double> pixelToPhysical(std::size_t i, std::size_t j, const SimulationParams& params) const;
    
    std::size_t findNearestCellPointBruteForce(double x, double y) const;
    std::size_t findNearestCellPointCoarseLists(double x, double y, const std::array<std::array<std::vector<std::size_t>, 16>, 16>& coarseCellCenterLists, const SimulationParams& params) const;
    std::size_t findCellContainingPointCoarseLists(double x, double y, const std::array<std::array<std::vector<std::size_t>,16>,16>& coarse, const SimulationParams& params) const;
    std::size_t findCellContainingPoint(double x, double y, const std::array<std::array<std::vector<std::size_t>, 16>, 16>& coarseCellCenterLists, const SimulationParams& params) const;
    void computePixelsToCellIDs(SimulationParams& params);

    inline void recordEdge(std::map<Key,Value>& edgeMap, std::size_t va, std::size_t vb, std::size_t cellIdx);

    void debug_dump_cells(const std::string& filename, double x_min_filter, double x_max_filter) const;

    explicit Mesh(const SimulationParams& params); // convertGmshToMesh
};


    
} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_MESH_HPP


