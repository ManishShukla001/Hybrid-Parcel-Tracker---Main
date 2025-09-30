#ifndef RK4_INTEGRATOR_H
#define RK4_INTEGRATOR_H

#include "interpolator.h"
#include <vector>
#include <array>

class RK4Integrator {
public:
    RK4Integrator(double dt_seconds, double data_interval_hours);
    
    ~RK4Integrator() = default;
    
    // Integrate a single particle using RK4 method
    std::array<double, 4> integrate_particle(
        double particle_id, double lat, double lon, double pressure,
        double alpha,
        const RegularGrid3DInterpolator& u_curr,
        const RegularGrid3DInterpolator& v_curr,
        const RegularGrid3DInterpolator& w_curr,
        const RegularGrid3DInterpolator& u_next,
        const RegularGrid3DInterpolator& v_next,
        const RegularGrid3DInterpolator& w_next,
        const std::array<double, 6>& bounds
    ) const;
    
    // Integrate a batch of particles
    void integrate_batch(
        const std::vector<std::array<double, 4>>& particles,
        std::vector<std::array<double, 4>>& results,
        double alpha,
        const RegularGrid3DInterpolator& u_curr,
        const RegularGrid3DInterpolator& v_curr,
        const RegularGrid3DInterpolator& w_curr,
        const RegularGrid3DInterpolator& u_next,
        const RegularGrid3DInterpolator& v_next,
        const RegularGrid3DInterpolator& w_next,
        const std::array<double, 6>& bounds
    ) const;
    
private:
    double dt_seconds_;
    double data_interval_hours_;
    
    // Helper function to get velocity at a point
    std::array<double, 3> get_velocity(
        double lat, double lon, double pressure,
        double alpha,
        const RegularGrid3DInterpolator& u_curr,
        const RegularGrid3DInterpolator& v_curr,
        const RegularGrid3DInterpolator& w_curr,
        const RegularGrid3DInterpolator& u_next,
        const RegularGrid3DInterpolator& v_next,
        const RegularGrid3DInterpolator& w_next,
        const std::array<double, 6>& bounds
    ) const;
};

#endif // RK4_INTEGRATOR_H