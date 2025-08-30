#include "renderer_colormap.hpp"




namespace navier_stokes {


//---------------------------------------------------------------------------
void render_scalar_field_unstructured(
        const Mesh& mesh,
        sf::Image& img,
        double fmin, double fmax,
        const std::string& cmap)
{
    const double span = fmax - fmin;
    if (span <= 0.0) return;

    for (const auto& triplet : mesh.pixelsToCellIDs)
    {
        const std::size_t  px     = triplet[0];
        const std::size_t  py     = triplet[1];
        const std::size_t  cid    = triplet[2];

        sf::Color col;
        if (cid == -1) {
            col = sf::Color::Black;
        }
        else
        {
            const double w = mesh.cells[cid].scalar;
            double       t = (w - fmin) / span;  // normalisation 0-1

            if (cmap == "schlieren") {
                t = std::sqrt(std::max(0.0, t));
                col = colormap_schlieren_bluewhite(t);
            } else {
                col = colormap_B_W_R(t);
            }
        }

        const unsigned ny = img.getSize().y;
        img.setPixel({static_cast<unsigned>(px),
                      static_cast<unsigned>(ny - 1 - py)},
                     col);
    }
}



void debugVisualizeScalarsUnstructured(
    Mesh& mesh,
    sf::Image& img,
    sf::Texture& tex,
    sf::Sprite& spr,
    sf::RenderWindow& win,
    double fmin,
    double fmax) {
    for (std::size_t cid = 0; cid < mesh.cells.size(); ++cid) {
        mesh.cells[cid].scalar = 0.0;
    }

    for (std::size_t cid = 0; cid < mesh.cells.size(); ++cid) {
        mesh.cells[cid].scalar = 1.0;

        render_scalar_field_unstructured(mesh, img, fmin, fmax, "");

        tex.update(img);
        spr.setTexture(tex, true);

        win.clear();
        win.draw(spr);
        win.display();

        while (auto ev = win.pollEvent()) {
            if (ev->is<sf::Event::Closed>()) {
                win.close();
                return; 
            }
        }
    }

    std::cerr << "Debug visualization complete, exiting.\n";
    std::exit(EXIT_FAILURE);
}


void debugVisualizeVertexContributions(
    Mesh& mesh,
    sf::Image& img,
    sf::Texture& tex,
    sf::Sprite& spr,
    sf::RenderWindow& win,
    double fmin,
    double fmax) { 
    for (auto& cell : mesh.cells) {
        cell.scalar = 0.0;
    }
 
    for (std::size_t vid = 0; vid < mesh.vertices.size(); ++vid) {
        const Vertex& vert = mesh.vertices[vid]; 
        for (std::size_t cid : vert.cellIDs) {
            double v = mesh.cells[cid].scalar + 1.0;
            mesh.cells[cid].scalar = std::clamp(v, 0.0, 4.0);
        }
 
        render_scalar_field_unstructured(mesh, img, fmin, fmax, "");

        tex.update(img);
        spr.setTexture(tex, true);

        win.clear();
        win.draw(spr);
        win.display();
 
        while (auto ev = win.pollEvent()) {
            if (ev->is<sf::Event::Closed>()) {
                win.close();
                return;
            }
        }
    }

    std::cerr << "Debug vertex contributions complete, exiting.\n";
    std::exit(EXIT_FAILURE);
}



std::pair<double,double> compute_display_size(
    double Lx, double Ly,
    double min_w, double min_h,
    double max_w, double max_h) {
    double ratio = Lx / Ly;
    double w, h;

    if (ratio >= 1.0) {
        // domaine plus large que haut
        h = min_h;
        w = ratio * h;
    } else {
        // domaine plus haut que large
        w = min_w;
        h = w / ratio;
    }

    // plafonner si necessaire
    if (w > max_w) {
        w = max_w;
        h = w / ratio;
    }
    if (h > max_h) {
        h = max_h;
        w = h * ratio;
    }
    return {w, h};
}



} // namespace navier_stokes



