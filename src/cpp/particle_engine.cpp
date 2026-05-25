// Enable math constants on Windows
#define _USE_MATH_DEFINES
#include "particle_engine.h"
#include "rk4_integrator.h"
#include "rk4_integrator_parallel.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <cstring>

namespace py = pybind11;

// Define M_PI if not available (Windows MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ParticleEngine::ParticleEngine(double dt_seconds, double data_interval_hours, bool use_parallel)
    : dt_seconds_(dt_seconds), data_interval_hours_(data_interval_hours) {
    if (use_parallel) {
        integrator_ = std::make_unique<ParallelRK4Integrator>(dt_seconds, data_interval_hours);
    } else {
        integrator_ = std::make_unique<RK4Integrator>(dt_seconds, data_interval_hours);
    }
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
    py::array_t<double> values,
    double fill_value
) {
    // Zero-copy access to values using unchecked proxy if possible, or request buffer
    py::buffer_info buf = values.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Values must be 1D array");
    }
    
    // Create vector from pointer (copying is necessary here as Interpolator owns the data in std::vector)
    // To truly avoid copy, Interpolator would need to manage a span or shared ptr, but that's a larger refactor.
    // However, we can at least use efficient copy construction.
    const double* ptr = static_cast<const double*>(buf.ptr);
    std::vector<double> val_vec(ptr, ptr + buf.size);

    return std::make_unique<RegularGrid3DInterpolator>(
        lat_coords, lon_coords, pressure_coords, val_vec, fill_value
    );
}

std::pair<py::array_t<double>, py::array_t<double>> ParticleEngine::update_particles(
    py::array_t<double> particles,
    double alpha,
    const RegularGrid3DInterpolator& u_curr,
    const RegularGrid3DInterpolator& v_curr,
    const RegularGrid3DInterpolator& w_curr,
    const RegularGrid3DInterpolator& u_next,
    const RegularGrid3DInterpolator& v_next,
    const RegularGrid3DInterpolator& w_next,
    const std::string& thermo_mode_str,
    const RegularGrid3DInterpolator* t_curr,
    const RegularGrid3DInterpolator* t_next,
    const RegularGrid3DInterpolator* t_mean_curr,
    const RegularGrid3DInterpolator* t_mean_next,
    const RegularGrid3DInterpolator* grad_t_lat_curr,
    const RegularGrid3DInterpolator* grad_t_lat_next,
    const RegularGrid3DInterpolator* grad_t_lon_curr,
    const RegularGrid3DInterpolator* grad_t_lon_next,
    const RegularGrid3DInterpolator* grad_t_p_curr,
    const RegularGrid3DInterpolator* grad_t_p_next,
    const RegularGrid3DInterpolator* dt_mean_dt_curr,
    const RegularGrid3DInterpolator* dt_mean_dt_next
) {
    py::buffer_info buf_in = particles.request();
    
    if (buf_in.ndim != 2 || buf_in.shape[1] != 4) {
        throw std::runtime_error("Particles must be (N, 4) array");
    }
    
    size_t count = buf_in.shape[0];
    const double* ptr_in = static_cast<const double*>(buf_in.ptr);
    
    // Allocate result array
    py::array_t<double> result = py::array_t<double>({static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(4)});
    py::buffer_info buf_out = result.request();
    double* ptr_out = static_cast<double*>(buf_out.ptr);
    
    ThermoMode thermo_mode = string_to_thermo_mode(thermo_mode_str);
    size_t thermo_cols = 0;
    if (thermo_mode == ThermoMode::SIMPLE_SUBTRACTION) thermo_cols = 1;
    else if (thermo_mode == ThermoMode::POTENTIAL_TEMPERATURE) thermo_cols = 2;
    else if (thermo_mode == ThermoMode::FULL_DECOMPOSITION) thermo_cols = 4;
    
    py::array_t<double> thermo_result = py::array_t<double>({static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(thermo_cols)});
    
    std::vector<ThermoState> thermo_states;
    ThermoState* thermo_ptr = nullptr;
    if (thermo_cols > 0) {
        thermo_states.resize(count);
        thermo_ptr = thermo_states.data();
    }
    
    ThermoInterpolators thermo_interp;
    thermo_interp.t_curr = t_curr;
    thermo_interp.t_next = t_next;
    thermo_interp.t_mean_curr = t_mean_curr;
    thermo_interp.t_mean_next = t_mean_next;
    thermo_interp.grad_t_lat_curr = grad_t_lat_curr;
    thermo_interp.grad_t_lat_next = grad_t_lat_next;
    thermo_interp.grad_t_lon_curr = grad_t_lon_curr;
    thermo_interp.grad_t_lon_next = grad_t_lon_next;
    thermo_interp.grad_t_p_curr = grad_t_p_curr;
    thermo_interp.grad_t_p_next = grad_t_p_next;
    thermo_interp.dt_mean_dt_curr = dt_mean_dt_curr;
    thermo_interp.dt_mean_dt_next = dt_mean_dt_next;

    auto bounds = u_curr.get_bounds();
    
    // Zero-copy integration!
    integrator_->integrate_batch(
        ptr_in, ptr_out, thermo_ptr, count, alpha,
        u_curr, v_curr, w_curr,
        u_next, v_next, w_next,
        bounds, thermo_mode, thermo_interp
    );
    
    // Copy thermo results into numpy array
    if (thermo_cols > 0) {
        py::buffer_info buf_thermo = thermo_result.request();
        double* ptr_thermo = static_cast<double*>(buf_thermo.ptr);
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < thermo_cols; ++j) {
                ptr_thermo[i * thermo_cols + j] = thermo_states[i].values[j];
            }
        }
    }
    
    return {result, thermo_result};
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
        .def(py::init<double, double, bool>(), py::arg("dt_seconds"), py::arg("data_interval_hours"), py::arg("use_parallel") = false)
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
             py::arg("u_next"), py::arg("v_next"), py::arg("w_next"),
             py::arg("thermo_mode_str") = "None",
             py::arg("t_curr") = nullptr, py::arg("t_next") = nullptr,
             py::arg("t_mean_curr") = nullptr, py::arg("t_mean_next") = nullptr,
             py::arg("grad_t_lat_curr") = nullptr, py::arg("grad_t_lat_next") = nullptr,
             py::arg("grad_t_lon_curr") = nullptr, py::arg("grad_t_lon_next") = nullptr,
             py::arg("grad_t_p_curr") = nullptr, py::arg("grad_t_p_next") = nullptr,
             py::arg("dt_mean_dt_curr") = nullptr, py::arg("dt_mean_dt_next") = nullptr)
        .def("get_coordinate_bounds", &ParticleEngine::get_coordinate_bounds);
}