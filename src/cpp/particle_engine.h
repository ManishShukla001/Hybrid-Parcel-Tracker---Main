#ifndef PARTICLE_ENGINE_H
#define PARTICLE_ENGINE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "rk4_integrator.h"
#include "interpolator.h"

namespace py = pybind11;

class ParticleEngine {
public:
    ParticleEngine(double dt_seconds, double data_interval_hours, bool use_parallel = false);
    
    ~ParticleEngine() = default;
    
    // Initialize particles on a regular grid
    std::vector<std::array<double, 4>> initialize_particles(
        double lat_start, double lat_end,
        double lon_start, double lon_end,
        const std::vector<double>& pressure_levels,
        double spacing_km
    );
    
    // Create interpolators from data arrays
    std::unique_ptr<RegularGrid3DInterpolator> create_interpolator(
        const std::vector<double>& lat_coords,
        const std::vector<double>& lon_coords,
        const std::vector<double>& pressure_coords,
        py::array_t<double> values,
        double fill_value = 0.0
    );
    
    // Update particle positions using RK4 integration
    // Returns a pair of numpy arrays: {updated_particles, delta_thermo_state}
    std::pair<py::array_t<double>, py::array_t<double>> update_particles(
        py::array_t<double> particles,
        double alpha,
        const RegularGrid3DInterpolator& u_curr,
        const RegularGrid3DInterpolator& v_curr,
        const RegularGrid3DInterpolator& w_curr,
        const RegularGrid3DInterpolator& u_next,
        const RegularGrid3DInterpolator& v_next,
        const RegularGrid3DInterpolator& w_next,
        const std::string& thermo_mode_str = "None",
        const RegularGrid3DInterpolator* t_curr = nullptr,
        const RegularGrid3DInterpolator* t_next = nullptr,
        const RegularGrid3DInterpolator* t_mean_curr = nullptr,
        const RegularGrid3DInterpolator* t_mean_next = nullptr,
        const RegularGrid3DInterpolator* grad_t_lat_curr = nullptr,
        const RegularGrid3DInterpolator* grad_t_lat_next = nullptr,
        const RegularGrid3DInterpolator* grad_t_lon_curr = nullptr,
        const RegularGrid3DInterpolator* grad_t_lon_next = nullptr,
        const RegularGrid3DInterpolator* grad_t_p_curr = nullptr,
        const RegularGrid3DInterpolator* grad_t_p_next = nullptr,
        const RegularGrid3DInterpolator* dt_mean_dt_curr = nullptr,
        const RegularGrid3DInterpolator* dt_mean_dt_next = nullptr
    );
    
    // Get coordinate bounds from interpolators
    std::array<double, 6> get_coordinate_bounds(
        const RegularGrid3DInterpolator& interpolator
    );
    
private:
    std::unique_ptr<RK4Integrator> integrator_;
    double dt_seconds_;
    double data_interval_hours_;
    
    // Helper function to calculate grid spacing
    double calculate_degree_spacing(double spacing_km, double latitude, bool is_longitude = false);
};

#endif // PARTICLE_ENGINE_H