// Enable math constants on Windows
#define _USE_MATH_DEFINES
#include "particle_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

// Define M_PI if not available (Windows MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ParticleEngine::ParticleEngine(double dt_seconds, double data_interval_hours)
    : dt_seconds_(dt_seconds), data_interval_hours_(data_interval_hours) {
    integrator_ = std::make_unique<RK4Integrator>(dt_seconds, data_interval_hours);
}

double ParticleEngine::calculate_degree_spacing(double spacing_km, double latitude, bool is_longitude) {
    const double deg_per_km_lat = 1.0 / 111.0;
    
    if (!is_longitude) {
        return spacing_km * deg_per_km_lat;
    } else {
        double lat_rad = latitude * M_PI / 180.0;
        double deg_per_km_lon = 1.0 / (111.0 * std::cos(lat_rad));
        return spacing_km * deg_per_km_lon;
    }
}

std::vector<std::array<double, 4>> ParticleEngine::initialize_particles(
    double lat_start, double lat_end,
    double lon_start, double lon_end,
    const std::vector<double>& pressure_levels,
    double spacing_km
) {
    // Calculate grid spacing
    double mid_lat = (lat_start + lat_end) / 2.0;
    double delta_lat = calculate_degree_spacing(spacing_km, mid_lat, false);
    double delta_lon = calculate_degree_spacing(spacing_km, mid_lat, true);
    
    std::vector<std::array<double, 4>> particles;
    
    double particle_id = 1.0;
    
    // Generate grid points
    // Ensure loops do not significantly overshoot the end points due to floating point arithmetic.
    // Add a small epsilon to ensure the last point is included if it's exactly at lat_end/lon_end.
    double epsilon = 1e-6; 
    for (double lat = lat_start; lat <= lat_end + epsilon; lat += delta_lat) {
        for (double lon = lon_start; lon <= lon_end + epsilon; lon += delta_lon) {
            for (double pressure : pressure_levels) {
                // Ensure we don't add particles outside the strict end bounds if lat/lon slightly overshot
                particles.push_back({particle_id, lat, lon, pressure});
                particle_id += 1.0;
            }
        }
    }
    
    return particles;
}

std::unique_ptr<RegularGrid3DInterpolator> ParticleEngine::create_interpolator(
    const std::vector<double>& lat_coords,
    const std::vector<double>& lon_coords,
    const std::vector<double>& pressure_coords,
    const std::vector<double>& values,
    double fill_value
) {
    return std::make_unique<RegularGrid3DInterpolator>(
        lat_coords, lon_coords, pressure_coords, values, fill_value
    );
}

std::vector<std::array<double, 4>> ParticleEngine::update_particles(
    const std::vector<std::array<double, 4>>& particles,
    double alpha,
    const RegularGrid3DInterpolator& u_curr,
    const RegularGrid3DInterpolator& v_curr,
    const RegularGrid3DInterpolator& w_curr,
    const RegularGrid3DInterpolator& u_next,
    const RegularGrid3DInterpolator& v_next,
    const RegularGrid3DInterpolator& w_next
) {
    std::vector<std::array<double, 4>> results;
    auto bounds = u_curr.get_bounds();
    
    integrator_->integrate_batch(
        particles, results, alpha,
        u_curr, v_curr, w_curr,
        u_next, v_next, w_next,
        bounds
    );
    
    return results;
}

std::array<double, 6> ParticleEngine::get_coordinate_bounds(
    const RegularGrid3DInterpolator& interpolator
) {
    return interpolator.get_bounds();
}

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(particle_engine_cpp, m) {
    m.doc() = "High-performance C++ particle tracking engine";
    
    py::class_<RegularGrid3DInterpolator>(m, "RegularGrid3DInterpolator")
        .def(py::init<const std::vector<double>&, const std::vector<double>&, 
                     const std::vector<double>&, const std::vector<double>&, double>(),
             py::arg("lat_coords"), py::arg("lon_coords"), py::arg("pressure_coords"),
             py::arg("values"), py::arg("fill_value") = 0.0)
        .def("interpolate", &RegularGrid3DInterpolator::interpolate)
        .def("interpolate_batch", &RegularGrid3DInterpolator::interpolate_batch)
        .def("get_bounds", &RegularGrid3DInterpolator::get_bounds);
    
    py::class_<ParticleEngine>(m, "ParticleEngine")
        .def(py::init<double, double>(), py::arg("dt_seconds"), py::arg("data_interval_hours"))
        .def("initialize_particles", &ParticleEngine::initialize_particles,
             py::arg("lat_start"), py::arg("lat_end"), py::arg("lon_start"), py::arg("lon_end"),
             py::arg("pressure_levels"), py::arg("spacing_km"))
        .def("create_interpolator", &ParticleEngine::create_interpolator,
             py::arg("lat_coords"), py::arg("lon_coords"), py::arg("pressure_coords"),
             py::arg("values"), py::arg("fill_value") = 0.0,
             py::return_value_policy::take_ownership)
        .def("update_particles", &ParticleEngine::update_particles,
             py::arg("particles"), py::arg("alpha"),
             py::arg("u_curr"), py::arg("v_curr"), py::arg("w_curr"),
             py::arg("u_next"), py::arg("v_next"), py::arg("w_next"))
        .def("get_coordinate_bounds", &ParticleEngine::get_coordinate_bounds);
}