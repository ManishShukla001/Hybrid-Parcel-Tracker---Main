#ifndef RK4_INTEGRATOR_PARALLEL_H
#define RK4_INTEGRATOR_PARALLEL_H

#include "rk4_integrator.h"

class ParallelRK4Integrator : public RK4Integrator {
public:
    ParallelRK4Integrator(double dt_seconds, double data_interval_hours)
        : RK4Integrator(dt_seconds, data_interval_hours) {}
    
    virtual ~ParallelRK4Integrator() = default;
    
    // Override integrate_batch to use OpenMP (Pointer version)
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
    ) const override;
};

#endif // RK4_INTEGRATOR_PARALLEL_H
