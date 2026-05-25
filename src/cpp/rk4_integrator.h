#ifndef RK4_INTEGRATOR_H
#define RK4_INTEGRATOR_H

#include "interpolator.h"
#include "thermo_state.h"
#include <vector>
#include <array>

class RK4Integrator {
public:
    RK4Integrator(double dt_seconds, double data_interval_hours);
    
    virtual ~RK4Integrator() = default;
    
    // Integrate a single particle using RK4 method
    // Returns pair of {new_position, delta_thermo_state}
    std::pair<std::array<double, 4>, ThermoState> integrate_particle(
        double particle_id, double lat, double lon, double pressure,
        double alpha,
        const RegularGrid3DInterpolator& u_curr,
        const RegularGrid3DInterpolator& v_curr,
        const RegularGrid3DInterpolator& w_curr,
        const RegularGrid3DInterpolator& u_next,
        const RegularGrid3DInterpolator& v_next,
        const RegularGrid3DInterpolator& w_next,
        const std::array<double, 6>& bounds,
        ThermoMode thermo_mode = ThermoMode::NONE,
        const ThermoInterpolators& thermo_interp = ThermoInterpolators()
    ) const;
    
    // Integrate a batch of particles (Vector version - helper)
    // This is not virtual anymore, it delegates to the pointer version
    void integrate_batch(
        const std::vector<std::array<double, 4>>& particles,
        std::vector<std::array<double, 4>>& results,
        std::vector<ThermoState>& thermo_results,
        double alpha,
        const RegularGrid3DInterpolator& u_curr,
        const RegularGrid3DInterpolator& v_curr,
        const RegularGrid3DInterpolator& w_curr,
        const RegularGrid3DInterpolator& u_next,
        const RegularGrid3DInterpolator& v_next,
        const RegularGrid3DInterpolator& w_next,
        const std::array<double, 6>& bounds,
        ThermoMode thermo_mode = ThermoMode::NONE,
        const ThermoInterpolators& thermo_interp = ThermoInterpolators()
    ) const;
    
    // Integrate a batch of particles (Pointer version - core implementation)
    // This is the virtual method that subclasses should override
    virtual void integrate_batch(
        const double* particles,
        double* results,
        ThermoState* thermo_results,
        size_t count,
        double alpha,
        const RegularGrid3DInterpolator& u_curr,
        const RegularGrid3DInterpolator& v_curr,
        const RegularGrid3DInterpolator& w_curr,
        const RegularGrid3DInterpolator& u_next,
        const RegularGrid3DInterpolator& v_next,
        const RegularGrid3DInterpolator& w_next,
        const std::array<double, 6>& bounds,
        ThermoMode thermo_mode = ThermoMode::NONE,
        const ThermoInterpolators& thermo_interp = ThermoInterpolators()
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