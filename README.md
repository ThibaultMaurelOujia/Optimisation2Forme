# Optimisation de forme par méthode adjointe

> ⚠️ **Avertissement**  
> Ce dépôt sert à expérimenter une boucle d'optimisation de forme par méthode adjointe. Le solveur d'écoulement est volontairement simple et diffus (maillages grossiers, schémas basiques, pas de modèle de turbulence avancé). Les champs de flux produits ici ne doivent pas être considérés comme des références quantitatives. L'objectif principal est la chaîne adjoint → gradient de forme → mise à jour géométrique, pas la haute fidélité aérodynamique. Pour des résultats quantitatifs fiables, il faut des maillages fins, des schémas d'ordre supérieur et un modèle de turbulence (RANS ou LES).


## Description de la simulation
- Navier–Stokes compressibles 2D (volumes finis).
- Gaz parfait ($\gamma$ constant).
- Viscosité constante.
- Maillage non structuré généré avec Gmsh.
- Schéma vertex-based (interpolation cellule → sommet → arête).
- Gradients par Green–Gauss.
- Limiteur de pente Venkatakrishnan (option pour Barth–Jespersen).
- Flux convectif HLLC.
- Avancement en temps RK3 SSP.
- Parallélisation OpenMP.

## Optimisation adjointe
- Adjoint discret sans assemblage explicite de matrice (matrix-free).
- Résolution de l'adjoint avec Eigen BiCGSTAB.
- Fonction objectif: $J = -C_L + \beta C_D$.
- Boucle : état direct → adjoint → gradient de forme → mise à jour géométrique → remeshing → ...

## Dépendances
- Eigen (algèbre linéaire).
- Gmsh (génération de maillages).
- OpenMP (parallélisme CPU).

