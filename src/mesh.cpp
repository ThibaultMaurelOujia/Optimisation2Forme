#include "mesh.hpp"

namespace navier_stokes {


// convertGmshToMesh
Mesh::Mesh(const SimulationParams& params) {

    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", 1);
    gmsh::open(params.mesh_name); // "naca_2D.msh"


    std::map<Key,Value> edgeMap;
    std::map<std::pair<std::size_t,std::size_t>, char> boundaryCondition;

    
    std::vector<std::size_t> nodeTags;
    std::vector<double> nodeCoords, nodeParametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParametricCoord);
    if (params.verbosity > 0)
        std::cout << "\n" << "Nombre de Noeuds : " << nodeTags.size() << "\n";

    this->vertices .reserve(nodeTags.size());
    this->vertexMap.reserve(nodeTags.size());

    double x_min =  std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    double y_min =  std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();
    for(std::size_t i = 0; i < nodeTags.size(); ++i) {
        double x = nodeCoords[3 * i + 0];
        double y = nodeCoords[3 * i + 1];
        double z = nodeCoords[3 * i + 2];
        size_t idx = this->vertices.size();
        this->vertexMap[nodeTags[i]] = idx;
        this->vertices.push_back(Vertex{vector_3D{x, y, z}}); 

        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);
        y_min = std::min(y_min, y);
        y_max = std::max(y_max, y);
    }
    this->x_min = x_min; this->x_max = x_max; this->y_min = y_min; this->y_max = y_max; 


    std::vector<std::pair<int,int>> physGroups;
    gmsh::model::getPhysicalGroups(physGroups);

    if (params.verbosity > 0)
        std::cout << "Nombre de groupes physiques : " << physGroups.size() << "\n";

    for(auto &pg : physGroups) {
        int dim = pg.first;
        int tag = pg.second;

        std::string name;
        gmsh::model::getPhysicalName(dim, tag, name);

        if (params.verbosity > 0)
                std::cout << "Groupe physique : dim=" << dim << ", tag=" << tag << ", nom=\"" << name << "\"\n";

        std::vector<int> entities;
        gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);

        std::size_t totalElems = 0;
        for(std::size_t eTag : entities) {
            std::vector<int> elemTypes;
            std::vector<std::vector<std::size_t>> elemTags, elemNodeTags;
            gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, dim, eTag);

            // https://victorsndvg.github.io/FEconv/formats/gmshmsh.xhtml 

            for(std::size_t i = 0; i<elemTypes.size(); ++i) {
                // std::cout << " elemTags : " << elemTags[i].size() << std::endl;
                if (elemTypes[i] == 1) {
                    for(std::size_t j = 0; j < elemTags[i].size(); ++j) {
                        size_t leftVertexID  = elemNodeTags[i][2 * j + 0];
                        size_t rightVertexID = elemNodeTags[i][2 * j + 1];

                        std::size_t vL, vR;
                        vL = std::min(this->vertexMap[leftVertexID], this->vertexMap[rightVertexID]);
                        vR = std::max(this->vertexMap[leftVertexID], this->vertexMap[rightVertexID]);
                        if (name == "Inlet") {
                            boundaryCondition[{vL, vR}] = 'i';
                            this->vertices[this->vertexMap[leftVertexID]].boundaryCondition = 'i';
                            this->vertices[this->vertexMap[rightVertexID]].boundaryCondition = 'i';
                        }
                        else if (name == "Outlet") {
                            boundaryCondition[{vL, vR}] = 'o';
                            this->vertices[this->vertexMap[leftVertexID]].boundaryCondition = 'o';
                            this->vertices[this->vertexMap[rightVertexID]].boundaryCondition = 'o';
                        }
                        else if (name == "FreeStream") {
                            boundaryCondition[{vL, vR}] = 'f';
                            this->vertices[this->vertexMap[leftVertexID]].boundaryCondition = 'f';
                            this->vertices[this->vertexMap[rightVertexID]].boundaryCondition = 'f';
                        }
                        else if (name == "Wall") {
                            boundaryCondition[{vL, vR}] = 'w';
                            this->vertices[this->vertexMap[leftVertexID]].boundaryCondition = 'w';
                            this->vertices[this->vertexMap[rightVertexID]].boundaryCondition = 'w';
                        }
                        else if (name == "Airfoil") {
                            boundaryCondition[{vL, vR}] = 'w';
                            this->vertices[this->vertexMap[leftVertexID]].boundaryCondition = 'w';
                            this->vertices[this->vertexMap[rightVertexID]].boundaryCondition = 'w';
                        }
                    }
                }
            }
            for(std::size_t i = 0; i<elemTypes.size(); ++i) {
                if (elemTypes[i] == 2) {
                    for(std::size_t j = 0; j < elemTags[i].size(); ++j) {
                        size_t n0ID = elemNodeTags[i][3 * j + 0];
                        size_t n1ID = elemNodeTags[i][3 * j + 1];
                        size_t n2ID = elemNodeTags[i][3 * j + 2];

                        size_t idx = this->cells.size();
                        // this->cellMap[elemTags[i][j]] = idx;
                        Cell cell;
                        // cell.verticeIDs = {n0ID, n1ID, n2ID};
                        cell.verticeIDs = {this->vertexMap[n0ID], this->vertexMap[n1ID], this->vertexMap[n2ID]};
                        vector_3D n0 = this->vertices[this->vertexMap[n0ID]].p;
                        vector_3D n1 = this->vertices[this->vertexMap[n1ID]].p;
                        vector_3D n2 = this->vertices[this->vertexMap[n2ID]].p;
                        vector_3D n0n1{n1.x - n0.x, n1.y - n0.y, 0};
                        vector_3D n0n2{n2.x - n0.x, n2.y - n0.y, 0};
                        cell.cellVolume = std::abs(0.5 * (n0n1.x * n0n2.y - n0n2.x * n0n1.y));
                        cell.centre  = vector_3D{1.0/3.0 * (n0.x + n1.x + n2.x), 1.0/3.0 * (n0.y + n1.y + n2.y), 0};

                        this->cells.push_back(cell); 
                        this->vertices[this->vertexMap[n0ID]].cellIDs.insert(idx);
                        this->vertices[this->vertexMap[n1ID]].cellIDs.insert(idx);
                        this->vertices[this->vertexMap[n2ID]].cellIDs.insert(idx);

                        recordEdge(edgeMap, vertexMap[n0ID], vertexMap[n1ID], idx);
                        recordEdge(edgeMap, vertexMap[n1ID], vertexMap[n2ID], idx);
                        recordEdge(edgeMap, vertexMap[n2ID], vertexMap[n0ID], idx);
                    }
                }
                else if (elemTypes[i] == 3) {
                    for(std::size_t j = 0; j < elemTags[i].size(); ++j) {
                        size_t n0ID = elemNodeTags[i][4 * j + 0];
                        size_t n1ID = elemNodeTags[i][4 * j + 1];
                        size_t n2ID = elemNodeTags[i][4 * j + 2];
                        size_t n3ID = elemNodeTags[i][4 * j + 3];

                        size_t idx = this->cells.size();
                        // this->cellMap[elemTags[i][j]] = idx;
                        Cell cell;
                        // cell.verticeIDs = {n0ID, n1ID, n2ID, n3ID};
                        cell.verticeIDs = {this->vertexMap[n0ID], this->vertexMap[n1ID], this->vertexMap[n2ID], this->vertexMap[n3ID]};
                        vector_3D n0 = this->vertices[this->vertexMap[n0ID]].p;
                        vector_3D n1 = this->vertices[this->vertexMap[n1ID]].p;
                        vector_3D n2 = this->vertices[this->vertexMap[n2ID]].p;
                        vector_3D n3 = this->vertices[this->vertexMap[n3ID]].p;
                        vector_3D n0n1{n1.x - n0.x, n1.y - n0.y, 0};
                        vector_3D n0n2{n2.x - n0.x, n2.y - n0.y, 0};
                        vector_3D n2n1{n1.x - n2.x, n1.y - n2.y, 0};
                        vector_3D n2n3{n3.x - n2.x, n3.y - n2.y, 0};
                        cell.cellVolume = 0.5 * (std::abs(n0n1.x * n0n2.y - n0n2.x * n0n1.y) + std::abs(n2n1.x * n2n3.y - n2n3.x * n2n1.y));
                        cell.centre = vector_3D{0.25 * (n0.x + n1.x + n2.x + n3.x), 0.25 * (n0.y + n1.y + n2.y + n3.y), 0};

                        this->cells.push_back(cell); 
                        this->vertices[this->vertexMap[n0ID]].cellIDs.insert(idx);
                        this->vertices[this->vertexMap[n1ID]].cellIDs.insert(idx);
                        this->vertices[this->vertexMap[n2ID]].cellIDs.insert(idx);
                        this->vertices[this->vertexMap[n3ID]].cellIDs.insert(idx);

                        recordEdge(edgeMap, vertexMap[n0ID], vertexMap[n1ID], idx);
                        recordEdge(edgeMap, vertexMap[n1ID], vertexMap[n2ID], idx);
                        recordEdge(edgeMap, vertexMap[n2ID], vertexMap[n3ID], idx);
                        recordEdge(edgeMap, vertexMap[n3ID], vertexMap[n0ID], idx);
                    }
                }
            }
        }
    }

    
    std::size_t edgeIdx = 0;
    for (const auto& [VertexIDLR, CellIDLR] : edgeMap) {
        const auto& [leftVertexID, rightVertexID] = VertexIDLR;
        const auto& [leftCellID, rightCellID]     = CellIDLR;

        Edge edge;
        edge.leftVertexID  = leftVertexID;
        edge.rightVertexID = rightVertexID;
        edge.leftCellID    = leftCellID;

        this->vertices[leftVertexID ].edgeIDs.insert(edgeIdx);
        this->vertices[rightVertexID].edgeIDs.insert(edgeIdx);

        auto itBC = boundaryCondition.find(VertexIDLR);
        if (itBC != boundaryCondition.end()) 
            edge.boundaryCondition = itBC->second;

        vector_3D leftVertex  = this->vertices[leftVertexID ].p;
        vector_3D rightVertex = this->vertices[rightVertexID].p;
        vector_3D edgeTangente = vector_3D{leftVertex.x - rightVertex.x, leftVertex.y - rightVertex.y, 0};
        edge.edgeLength = std::sqrt(edgeTangente.x * edgeTangente.x + edgeTangente.y * edgeTangente.y + edgeTangente.z * edgeTangente.z);
        edge.edgeNormal = vector_3D{edgeTangente.y / edge.edgeLength, - edgeTangente.x / edge.edgeLength, 0};
        
        vector_3D centreLeftCell  = this->cells[leftCellID].centre;
        vector_3D centreEdge = vector_3D{0.5*(this->vertices[leftVertexID].p.x+this->vertices[rightVertexID].p.x), 0.5*(this->vertices[leftVertexID].p.y+this->vertices[rightVertexID].p.y), 0};
        vector_3D vectorCellLR{centreLeftCell.x - centreEdge.x, centreLeftCell.y - centreEdge.y, 0};
        if (vectorCellLR.x * edge.edgeNormal.x + vectorCellLR.y * edge.edgeNormal.y < 0)
            edge.edgeNormal = vector_3D{-edge.edgeNormal.x, -edge.edgeNormal.y, 0};
        
        edge.centre = centreEdge;

        this->cells[leftCellID].edgeIDs.insert(edgeIdx);

        if (rightCellID != -1) {
            edge.rightCellID = rightCellID;

            this->cells[rightCellID].edgeIDs.insert(edgeIdx);

            this->cells[leftCellID ].neighbourIDs.insert(rightCellID);
            this->cells[rightCellID].neighbourIDs.insert(leftCellID );
        }
        else 
            this->cells[leftCellID].border = true;
        
        this->edges.push_back(edge);

        ++edgeIdx;
    }

    for(std::size_t vid = 0; vid < this->vertices.size(); ++vid) {
        auto &vert = this->vertices[vid];
        // recuper (angle, cid)
        std::vector<std::pair<double,std::size_t>> tmp;
        tmp.reserve(vert.cellIDs.size());
        for(auto cid : vert.cellIDs) {
            const auto &C = this->cells[cid].centre;
            double ang = std::atan2(C.y - vert.p.y, C.x - vert.p.x);
            tmp.emplace_back(ang, cid);
        }
        // tri croissant angle
        std::sort(tmp.begin(), tmp.end(),
                [](auto &a, auto &b){ return a.first < b.first; });
        // remplit l'ordre
        vert.cellIDsOrdered.clear();
        vert.cellIDsOrdered.reserve(tmp.size());
        for(auto &pr : tmp) vert.cellIDsOrdered.push_back(pr.second);
    }


    // debug_dump_cells("debug_cells.txt", 1.5, 2.0);


{
    std::ofstream ofs("debug_cells_edges.txt");
    if(!ofs) {
        std::cerr << "Erreur debug_cells_edges.txt\n";
    } else {
        ofs << std::fixed << std::setprecision(6);
        for(std::size_t cid = 0; cid < this->cells.size(); ++cid) {
            const auto& cell = this->cells[cid];
            
            ofs << cell.centre.x << "," << cell.centre.y;
            
            for(auto eid : cell.edgeIDs) {
                const auto& e = this->edges[eid];
                const auto& pA = this->vertices[e.leftVertexID].p;
                const auto& pB = this->vertices[e.rightVertexID].p;
                double mx = 0.5*(pA.x + pB.x);
                double my = 0.5*(pA.y + pB.y);
                ofs << "," << mx << "," << my;
            }
            ofs << "\n";
        }
        ofs.close();
        std::cout << "debug_cells_edges.txt\n";
    }
}


    gmsh::clear();
    gmsh::finalize();
}




inline void Mesh::recordEdge(std::map<Key,Value>& edgeMap, std::size_t va, std::size_t vb, std::size_t cellIdx) {
    // Key k{va, vb};
    Key k{ std::min(va, vb), std::max(va, vb) };
    auto it = edgeMap.find(k);
    if(it == edgeMap.end()) {
        it = edgeMap.emplace(k, Value{ cellIdx, -1 }).first;
    } else {
        it->second.second = cellIdx;
    }
}




void Mesh::debug_dump_cells(const std::string& filename, double x_min_filter, double x_max_filter) const {
    std::ofstream ofs(filename);
    if(!ofs) {
        std::cerr << "Erreur ouverture " << filename << "\n";
        return;
    }
    ofs << std::fixed << std::setprecision(6);
    ofs << "// cid, cx, cy,  [vid:(vx,vy)...],  [eid...],  [neigh...]\n";
    for(std::size_t cid = 0; cid < cells.size(); ++cid) {
        const Cell& C = cells[cid];
        double cx = C.centre.x, cy = C.centre.y;
        if(cx <= x_min_filter || cx >= x_max_filter) continue;

        // début de ligne
        ofs << cid << ", " << cx << ", " << cy;

        // sommets
        ofs << ", verts{";
        for(std::size_t vid : C.verticeIDs) {
            const auto& P = vertices[vid].p;
            ofs << vid << ":(" << P.x << "," << P.y << ") ";
        }
        ofs << "}";

        // aretes
        ofs << ", edges{";
        for(std::size_t eid : C.edgeIDs) {
            ofs << eid << " ";
        }
        ofs << "}";

        // voisins
        ofs << ", neigh{";
        for(std::size_t nid : C.neighbourIDs) {
            ofs << nid << " ";
        }
        ofs << "}";

        ofs << "\n";
    }
    ofs.close();
    std::cout << "Écriture " << filename << " terminée\n";
}




std::pair<double,double> Mesh::pixelToPhysical(std::size_t i, std::size_t j, const SimulationParams& params) const {
    double x = x_min + (i + 0.5) * params.Lx / params.Nx;
    double y = y_min + (j + 0.5) * params.Ly / params.Ny;
    return {x, y};
}

std::size_t Mesh::findNearestCellPointBruteForce(double x, double y) const {
    std::size_t min_distance_cell_ID = 0;
    double min_distance = std::numeric_limits<double>::infinity();
    for(std::size_t i = 0; i<this->cells.size(); ++i) {
        const Cell& cell = this->cells[i];
        double dx = cell.centre.x - x;
        double dy = cell.centre.y - y;
        double distance = dx * dx + dy * dy;
        if (distance < min_distance) {
            min_distance = distance;
            min_distance_cell_ID = i;
        }
    }
    return min_distance_cell_ID;
}

std::size_t Mesh::findNearestCellPointCoarseLists(double x, double y, const std::array<std::array<std::vector<std::size_t>, 16>, 16>& coarseCellCenterLists, const SimulationParams& params) const {
    std::size_t min_distance_cell_ID = -1;
    double min_distance = std::numeric_limits<double>::infinity();
    int ix = std::min(int(std::floor((x - this->x_min) / params.Lx * 16)), 16-1);
    int iy = std::min(int(std::floor((y - this->y_min) / params.Ly * 16)), 16-1);
    
    for(int i = -1; i < 2; ++i) {
        for(int j = -1; j < 2; ++j) {
            int ixn = ix + i; int iyn = iy + j;  
            if (0 <= ixn && ixn < 16 && 0 <= iyn && iyn < 16) {
                for(std::size_t k : coarseCellCenterLists[ixn][iyn]) {
                    const Cell& cell = this->cells[k];
                    double dx = cell.centre.x - x;
                    double dy = cell.centre.y - y;
                    double distance = dx * dx + dy * dy;
                    if (distance < min_distance) {
                        min_distance = distance;
                        min_distance_cell_ID = k;
                        if (cell.border)
                            min_distance_cell_ID = -1;
                    }
                }
            }
        }
    }
    // if (min_distance_cell_ID == -1)
    //     return findCellContainingPointBruteForce(x, y);
    return min_distance_cell_ID;
}


static inline bool pointInTriangle(double x, double y, const vector_3D& v0, const vector_3D& v1, const vector_3D& v2) {
    double  dX  = x - v2.x,  dY  = y - v2.y;
    double  dX21= v2.x - v1.x, dY12 = v1.y - v2.y;
    double  D   = dY12*(v0.x-v2.x) + dX21*(v0.y-v2.y);

    double  s   = dY12*dX + dX21*dY;
    double  t   = (v2.y-v0.y)*dX + (v0.x-v2.x)*dY;

    if (D < 0) { s = -s; t = -t; D = -D; }
    return (s >= 0 && t >= 0 && (s+t) <= D);
}


std::size_t Mesh::findCellContainingPointCoarseLists(
        double x, double y,
        const std::array<std::array<std::vector<std::size_t>,16>,16>& coarse,
        const SimulationParams& params) const
{
    int ix = std::min(int(std::floor((x - x_min)/params.Lx * 16)), 15);
    int iy = std::min(int(std::floor((y - y_min)/params.Ly * 16)), 15);

    std::size_t best = std::size_t(-1);
    double      best_d2 = std::numeric_limits<double>::infinity();

    static constexpr int off[3] = { 0, -1, 1 };
    for (int oi = 0; oi < 3; ++oi) {
        for (int oj = 0; oj < 3; ++oj) {
            int di = off[oi];
            int dj = off[oj];
    // for (int di=-1; di<=1; ++di)
    //     for (int dj=-1; dj<=1; ++dj)
    //     {
            int cx = ix + di, cy = iy + dj;
            if (cx<0 || cx>15 || cy<0 || cy>15) continue;

            for (std::size_t cid : coarse[cx][cy]) {
                const Cell& cell = cells[cid];
                // if (cell.border) continue; 

                const auto& vList = cell.verticeIDs;
                if (vList.size() == 3) {
                    const vector_3D &A = vertices[vList[0]].p, &B = vertices[vList[1]].p, &C= vertices[vList[2]].p;
                    if (pointInTriangle(x,y,A,B,C)) {
                        // if (cell.border)
                        //     return -1;
                        return cid; // trouver
                    } 
                }
                else if (vList.size() == 4) {
                    const vector_3D &A = vertices[vList[0]].p, &B = vertices[vList[1]].p, &C= vertices[vList[2]].p, &D = vertices[vList[3]].p;
                    if (pointInTriangle(x,y,A,B,C) || pointInTriangle(x,y,A,C,D)) {
                        // if (cell.border)
                        //     return -1;
                        return cid; // trouver
                    } 
                }

                // on garde le plus proche sinon
                double dx = cell.centre.x - x;
                double dy = cell.centre.y - y;
                double d2 = dx*dx + dy*dy;
                if (d2 < best_d2) { 
                    best_d2 = d2; best = cid; 
                    if (cell.border)
                        best = -1;
                }
            }
        }
    }

    return best; 
}


std::size_t Mesh::findCellContainingPoint(double x, double y, const std::array<std::array<std::vector<std::size_t>, 16>, 16>& coarseCellCenterList, const SimulationParams& params) const {
    // return findNearestCellPointBruteForce(x, y);
    // return findNearestCellPointCoarseLists(x, y, coarseCellCenterList, params);
    return findCellContainingPointCoarseLists(x, y, coarseCellCenterList, params);
}

void Mesh::computePixelsToCellIDs(SimulationParams& params) {
    auto t_pixel = time_tic();
    // Le domaine est rectangulaire
    params.Lx = std::abs(x_max - x_min);
    params.Ly = std::abs(y_max - y_min);


    std::array<std::array<std::vector<std::size_t>, 16>, 16> coarseCellCenterLists;
    for(std::size_t i = 0; i<this->cells.size(); ++i){
        Cell cell = this->cells[i];
        std::size_t ix = std::min(int(std::floor((cell.centre.x - x_min) / params.Lx * 16)), 16-1);
        std::size_t iy = std::min(int(std::floor((cell.centre.y - y_min) / params.Ly * 16)), 16-1);
        coarseCellCenterLists[ix][iy].push_back(i);
    }

    // pixelsToCellIDs.clear();
    // pixelsToCellIDs.reserve(params.Nx * params.Ny);
    pixelsToCellIDs.resize(params.Nx * params.Ny);
    #pragma omp parallel for
    for (std::size_t j = 0; j < params.Ny; ++j) {
        for (std::size_t i = 0; i < params.Nx; ++i) { 
            auto [x, y] = pixelToPhysical(i, j, params); // std::pair<double,double> 
            std::size_t cellID = findCellContainingPoint(x, y, coarseCellCenterLists, params);
            // pixelsToCellIDs.push_back({i, j, cellID});
            std::size_t idx = j * params.Nx + i;
            pixelsToCellIDs[idx] = { i, j, cellID };
        }
    }
    time_toc(t_pixel, "Pixel->Cell mapping", params.verbosity); 
}










} // namespace navier_stokes






    // for (std::size_t idx : this->vertices[0].edgeIDs) {
    //     std::cout << " idx : " << idx << std::endl;
    // }

    // // const auto& cell0 = this->cells[0];
    // const auto& cell0 = this->cells[100];
    // std::cout << "Cell 0 has vertices (IDs): ";
    // for (std::size_t vid : cell0.verticeIDs) {
    //     std::cout << vid << " ";
    // }
    // std::cout << "\n\n";

    // for (std::size_t vid : cell0.verticeIDs) {
    //     const auto& vert = this->vertices[vid];
    //     std::cout << "Vertex " << vid << " is in cells: ";
    //     for (std::size_t cid : vert.cellIDs) {
    //         std::cout << cid << " ";
    //     }
    //     std::cout << "\n";

    //     std::cout << "  Edges of vertex " << vid << ": ";
    //     for (std::size_t eid : vert.edgeIDs) {
    //         const auto& e = this->edges[eid];
    //         bool isBoundary = (e.boundaryCondition != '\0');
    //         std::cout
    //             << "[eid=" << eid
    //             << ", BC=" << (e.boundaryCondition ? e.boundaryCondition : '-')
    //             << ", boundary=" << (isBoundary ? "yes" : "no")
    //             << "] ";
    //     }
    //     std::cout << "\n\n";
    // }

    // std::cout << std::flush;






// {
//     std::ofstream ofs("debug_cells_edges.txt");
//     if(!ofs) {
//         std::cerr << "Erreur debug_cells_edges.txt\n";
//     } else {
//         ofs << std::fixed << std::setprecision(6);
//         for(std::size_t cid = 0; cid < this->cells.size(); ++cid) {
//             const auto& cell = this->cells[cid];
            
//             ofs << cell.centre.x << "," << cell.centre.y;
            
//             for(auto eid : cell.edgeIDs) {
//                 const auto& e = this->edges[eid];
//                 const auto& pA = this->vertices[e.leftVertexID].p;
//                 const auto& pB = this->vertices[e.rightVertexID].p;
//                 double mx = 0.5*(pA.x + pB.x);
//                 double my = 0.5*(pA.y + pB.y);
//                 ofs << "," << mx << "," << my;
//             }
//             ofs << "\n";
//         }
//         ofs.close();
//         std::cout << "debug_cells_edges.txt\n";
//     }
// }

























