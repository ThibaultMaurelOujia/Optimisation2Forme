cmake_minimum_required(VERSION 3.10)
project(NS_2D_compressible VERSION 1.0 LANGUAGES CXX)

# -- C++ standard --
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Debug or Release)" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling AddressSanitizer & UBSan in Debug")
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS_DEBUG
        "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()



# -- Répertoires include --
# Ajout du dossier include/ au chemin de recherche des headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# -- Rassembler tous les .cpp de src/ --
# Recherche récursive de tous les fichiers .cpp sous src/
file(GLOB_RECURSE NS_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# -- Exécutable principal --
# Déclaration de l’exécutable principal à partir des sources collectées
add_executable(navier_stokes
    ${NS_SOURCES}
)
set_target_properties(navier_stokes PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Ajout des options ASan/UBSan sur la cible navier_stokes")
  target_compile_options(navier_stokes PRIVATE
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
    -g
  )
  target_link_options(navier_stokes PRIVATE
    -fsanitize=address,undefined
  )
endif()






if(APPLE)
  # détecte automatiquement le préfix Homebrew pour libomp
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # dossiers d'include et de bibliothèque pour libomp
  set(OMP_INC "${LIBOMP_PREFIX}/include")
  set(OMP_LIB "${LIBOMP_PREFIX}/lib/libomp.dylib")

  # on ajoute omp.h
  target_include_directories(navier_stokes PUBLIC ${OMP_INC})

  # indique à clang d'utiliser OpenMP au pré-processing
  target_compile_options(navier_stokes PUBLIC
    "-Xpreprocessor" "-fopenmp"
  )

  # on linke la runtime OpenMP
  target_link_libraries(navier_stokes PUBLIC ${OMP_LIB})
endif()




# -- EIGEN --
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(navier_stokes PRIVATE Eigen3::Eigen)



# -- SFML --
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)
target_link_libraries(navier_stokes PRIVATE
  SFML::Graphics
  SFML::Window
  SFML::System
)



# -- GMSH --
#list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/lib/cmake/gmsh")
#find_package(Gmsh REQUIRED)
#target_link_libraries(navier_stokes PRIVATE Gmsh::gmsh)
find_path(GMSH_INCLUDE_DIR
  NAMES gmsh.h
  HINTS /opt/homebrew/opt/gmsh/include
)
find_library(GMSH_LIBRARY
  NAMES gmsh gmsh.4.13.1
  HINTS /opt/homebrew/opt/gmsh/lib
)
if (GMSH_INCLUDE_DIR AND GMSH_LIBRARY)
  message(STATUS "Gmsh trouvé : ${GMSH_INCLUDE_DIR}, ${GMSH_LIBRARY}")
  target_include_directories(navier_stokes PRIVATE ${GMSH_INCLUDE_DIR})
  target_link_libraries(navier_stokes PRIVATE ${GMSH_LIBRARY})
else ()
  message(FATAL_ERROR "Impossible de trouver Gmsh (headers ou lib).")
endif()


# -- Cible rebuild -- 
add_custom_target(rebuild
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMENT "Rebuild entire project"
)

# -------------------------------------------------------------------
# TODO: activer plus tard l’installation de l’exécutable
# install(TARGETS navier_stokes
#     RUNTIME DESTINATION bin
# )
# -------------------------------------------------------------------

# -- Pour afficher la liste des cibles personnalisées --
set_property(GLOBAL PROPERTY RULE_LAUNCH_HELP TRUE)



# cmake -DCMAKE_BUILD_TYPE=Debug

# make all
# make clean
# make rebuild
# make help
# make install
