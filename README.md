# Optimisation de forme par méthode adjointe — prototype

> **Disclaimer.** Ce dépôt sert à expérimenter une boucle d’optimisation de forme par méthode adjointe. Le solveur d’écoulement est volontairement simple et diffus (maillages grossiers, schémas basiques, pas de modèle de turbulence avancé). Les champs de flux produits ici ne doivent pas être considérés comme des références quantitatives. L’objectif principal est la chaîne adjoint → gradient de forme → mise à jour géométrique, pas la haute fidélité aérodynamique. Pour de la précision, il faudrait des maillages fins, des schémas d’ordre supérieur, et éventuellement LES/RANS.

## Description rapide
- Navier–Stokes compressibles 2D (volumes finis).
- Gaz parfait avec $\\gamma$ constant.
- Viscosité constante.
- Maillage Gmsh non structuré.
- Schéma vertex-based.
- Gradients: Green--Gauss.
- Limiteur: Venkatakrishnan (option pour Barth–Jespersen).
- Flux convectif: HLLC.
- Avancement en temps: RK3 SSP.
- Parallélisation: OpenMP.

## Optimisation adjointe
- Adjoint discret.
- Résolution adjoint: Eigen BiCGSTAB.
- Fonction objectif: $J = -C_L + \beta C_D$.
- Adjoint → gradient de forme → mise à jour géométrique.

## Portée et limitations
- Code à visée pédagogique.
- Résultats d’écoulement qualitatifs uniquement.
- Pas de turbulence (ni RANS ni LES).
- Diffusion numérique notable et maillages volontairement grossiers.

## Dépendances
- Eigen (algèbre linéaire).
- Gmsh (génération de maillage).
- OpenMP (parallélisme CPU).
