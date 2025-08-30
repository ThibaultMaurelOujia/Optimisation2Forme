#ifndef NAVIER_STOKES_CPP_COLORMAP_HPP
#define NAVIER_STOKES_CPP_COLORMAP_HPP

#include "utils.hpp"
#include "mesh.hpp"
#include "boundary.hpp"

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Image.hpp> 

#include <cstdint> 

namespace navier_stokes {

inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

// t in [0,1]  
inline sf::Color colormap_B_W_R(double t_)
{
    // clamp
    double t = std::max(0.0, std::min(1.0, t_));

    // bleu :
    // 0.0 -> 0.3, 0.25 -> 1.0, 0.5 -> 1.0, 0.75 -> 0.0, 1.0 -> 0.0
    double b;
    if (t < 0.25) {
        b = lerp(0.3, 1.0, t / 0.25);
    } else if (t < 0.5) {
        b = 1.0;
    } else if (t < 0.75) {
        b = lerp(1.0, 0.0, (t - 0.5) / 0.25);
    } else {
        b = 0.0;
    }

    // vert :
    // 0.0 -> 0.0, 0.25 -> 0.0, 0.5 -> 1.0, 0.75 -> 0.0, 1.0 -> 0.0
    double g;
    if (t < 0.25) {
        g = 0.0;
    } else if (t < 0.5) {
        g = lerp(0.0, 1.0, (t - 0.25) / 0.25);
    } else if (t < 0.75) {
        g = lerp(1.0, 0.0, (t - 0.5) / 0.25);
    } else {
        g = 0.0;
    }

    // rouge :
    // 0.0 -> 0.0, 0.25 -> 0.0, 0.5 -> 1.0, 0.75 -> 1.0, 1.0 -> 0.5
    double r;
    if (t < 0.25) {
        r = 0.0;
    } else if (t < 0.5) {
        r = lerp(0.0, 1.0, (t - 0.25) / 0.25);
    } else if (t < 0.75) {
        r = 1.0;
    } else {
        r = lerp(1.0, 0.5, (t - 0.75) / 0.25);
    }

    // passage en 0-255
    auto to8 = [&](double x){
        return static_cast<std::uint8_t>(std::max(0.0, std::min(1.0, x)) * 255.0 + 0.5);
    };

    return sf::Color{ to8(r), to8(g), to8(b) };
}


inline sf::Color colormap_schlieren_bluewhite(double t_)
{
    // clamp
    double t = std::max(0.0, std::min(1.0, t_));

    uint8_t r, g, b;

    if (t < 0.7) {                      // phase Noir -> Bleu pur
        double s = t / 0.7;             // 0..1
        r = 0;
        g = 0;
        b = static_cast<uint8_t>(lerp(0.0, 255.0, s) + 0.5);
    }
    else {                              // phase Bleu pur -> Blanc
        double s = (t - 0.7) / 0.3;     // 0..1
        r = static_cast<uint8_t>(lerp(0.0, 255.0, s) + 0.5);
        g = static_cast<uint8_t>(lerp(0.0, 255.0, s) + 0.5);
        b = 255;
    }

    return sf::Color{ r, g, b };
}


void render_scalar_field_unstructured(
        const Mesh&    mesh,
        sf::Image&     img,
        double         fmin,
        double         fmax,
        const std::string& cmap = "");
    
void debugVisualizeScalarsUnstructured(
    Mesh& mesh,
    sf::Image& img,
    sf::Texture& tex,
    sf::Sprite& spr,
    sf::RenderWindow& win,
    double fmin = 0.0,
    double fmax = 1.0);

void debugVisualizeVertexContributions(
    Mesh& mesh,
    sf::Image& img,
    sf::Texture& tex,
    sf::Sprite& spr,
    sf::RenderWindow& win,
    double fmin = 0.0,
    double fmax = 4.0);


std::pair<double,double> compute_display_size(
    double Lx, double Ly,
    double min_w = 7.0, double min_h = 8.0,
    double max_w = 15.0, double max_h = 10.0);



} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_COLORMAP_HPP