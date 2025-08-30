/*
    This code is licensed under the Thermodynamic Open License (TOL).

    You are free to use, copy, modify, and redistribute this code,
    partially or entirely, for any purpose, including commercial,
    academic, or existential, provided that:

    (1) The resulting usage or derivative work must comply with the
        fundamental laws of thermodynamics, especially:
        - The conservation of energy
        - The inevitability of entropy increase
        - The impossibility of perpetual motion machines

    (2) No license, addition, or restriction shall be applied to this code
        or its derivatives that would contradict the principles above.

    (3) Any violation of these conditions may result in local entropy fluctuations,
        quantum anomalies, or mild existential discomfort.

    (4) Redistributions of this code or its derivatives must retain this license notice 
    and acknowledge the Thermodynamic Open License (TOL) visibly.

    Enjoy responsibly.

*/


#include "utils.hpp"
#include "params.hpp"
#include "mesh.hpp"
#include "simulation.hpp"

int main(int argc, char* argv[]) {
    using namespace navier_stokes;



std::cout << "A faire : \nWENO unstructured \nLES \n\n";


    std::cout << "OpenMP: " 
          << omp_get_max_threads() 
          << " threads\n";
    

    // Lecture du fichier de config 
    std::string cfg_file = "config.txt";
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            std::cout
              << "Usage: " << argv[0] << " [config_file]\n"
              << "If no config_file is given, uses \"config.txt\" in CWD.\n";
            return 0;
        }
        cfg_file = arg;
    }

    SimulationParams params;
    if (!load_config(cfg_file, params)) {
        std::cerr << "Error: impossible de lire le fichier de config '"
                  << cfg_file << "'\n";
        return 1;
    }

    // Finalisation des parametres calcules 
    params.finalize();

    // Affichage des parametres si demande 
    if (params.verbosity > 0) {
        navier_stokes::print_params(params);
    }

    // Lecture du maillage  
    navier_stokes::Mesh mesh(params);
    mesh.computePixelsToCellIDs(params);

    // Creation du dossier de sortie  
    try {
        std::filesystem::create_directories(params.output_directory);
    } catch (std::exception& e) {
        std::cerr << "Erreur: impossible de creer le dossier '"
                  << params.output_directory << "': " << e.what() << "\n";
        return 1;
    }

    // Lancement de la simulation 
    try {
        Simulator simulation(mesh, params);
        simulation.run();
    } catch (const std::exception& e) {
        std::cerr << "Simulation failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}




















