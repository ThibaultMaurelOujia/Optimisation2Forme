#include "utils.hpp"
#include "boundary.hpp"


namespace navier_stokes {






//---------------------------------------------------------------------------
TimePoint time_tic() {
    return Clock::now();
}

double time_toc(const TimePoint& t0, const std::string& msg, int verbosity) {
    auto now     = Clock::now();
    double elapsed = std::chrono::duration<double>(now - t0).count();
    if (verbosity >= 2 && !msg.empty())
        std::cout << "[Timer] " << msg << ": " << elapsed << " s\n";
    return elapsed;
}
// time_tic();
// time_toc("Poisson direct 1"); 



void loadCellStates(Mesh& mesh, const SimulationParams& params) {
    const std::string& filepath = params.state_load_path;

    if (!filepath.empty() && filepath != "none") {
        std::ifstream in(filepath);
        if (!in) {
            throw std::runtime_error("Impossible d'ouvrir " + filepath + " en lecture.");
        }

        std::size_t Nfile = 0;
        in >> Nfile;
        if (!in) {
            throw std::runtime_error("Fichier corrompu (" + filepath + "): impossible de lire le nombre de cellules.");
        }

        if (Nfile != mesh.cells.size()) {
            std::ostringstream oss;
            oss << "Taille incohérente (" << filepath << "): fichier=" << Nfile
                << " vs mesh=" << mesh.cells.size()
                << " (vérifier l'ordre/la topologie).";
            throw std::runtime_error(oss.str());
        }

        for (std::size_t i = 0; i < Nfile; ++i) {
            double rho, ru, rv, E;
            in >> rho >> ru >> rv >> E;
            if (!in) {
                std::ostringstream oss;
                oss << "Lecture interrompue à la cellule " << i << " dans " << filepath << ".";
                throw std::runtime_error(oss.str());
            }
            Cell& c = mesh.cells[i];
            c.rho   = rho;
            c.rho_u = ru;
            c.rho_v = rv;
            c.E     = E;
        }
    }
}


void saveCellStates(const Mesh& mesh, const SimulationParams& params) {
    const std::string& filepath = params.state_save_path;

    if (!filepath.empty() && filepath != "none") {
        namespace fs = std::filesystem;

        // Crée les dossiers si besoin 
        fs::path p(filepath);
        if (!p.parent_path().empty()) {
            std::error_code ec;
            if (!fs::create_directories(p.parent_path(), ec) && ec) {
                throw std::runtime_error("Impossible de créer le dossier: " + p.parent_path().string());
            }
        }

        // Ouvre le fichier en écriture  
        std::ofstream out(filepath);
        if (!out) {
            throw std::runtime_error("Impossible d'ouvrir " + filepath + " en écriture.");
        }
 
        out << mesh.cells.size() << '\n';
        out << std::setprecision(17) << std::scientific; // précision élevée et format stable

        for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
            const Cell& c = mesh.cells[i];
            out << c.rho   << ' '
                << c.rho_u << ' '
                << c.rho_v << ' '
                << c.E     << '\n';
        }
 
        out.flush();
        if (!out) {
            throw std::runtime_error("Erreur d'écriture dans " + filepath);
        }
    }
}


} // namespace navier_stokes