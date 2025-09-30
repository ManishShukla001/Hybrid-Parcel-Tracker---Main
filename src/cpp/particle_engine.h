#ifndef PARTICLE_ENGINE_H
#define PARTICLE_ENGINE_H

#include "interpolator.h"
#include "rk4_integrator.h"
#include <vector>
#include <array>
#include <memory>

class ParticleEngine {
public:
    ParticleEngine(double dt_seconds, double data_interval_hours);
    
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
        const std::vector<double>& values,
        double fill_value = 0.0
    );
    
    // Update particle positions using RK4 integration
    std::vector<std::array<double, 4>> update_particles(
        const std::vector<std::array<double, 4>>& particles,
        double alpha,
        const RegularGrid3DInterpolator& u_curr,
        const RegularGrid3DInterpolator& v_curr,
        const RegularGrid3DInterpolator& w_curr,
        const RegularGrid3DInterpolator& u_next,
        const RegularGrid3DInterpolator& v_next,
        const RegularGrid3DInterpolator& w_next
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