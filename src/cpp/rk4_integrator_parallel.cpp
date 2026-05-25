// Enable math constants on Windows
#define _USE_MATH_DEFINES
#include "rk4_integrator_parallel.h"
#include <omp.h>
#include <iostream>

void ParallelRK4Integrator::integrate_batch(
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
    ThermoMode thermo_mode,
    const ThermoInterpolators& thermo_interp
) const {
    // Use OpenMP to parallelize the loop
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(count); ++i) {
        const double* p = particles + i * 4;
        double* r = results + i * 4;
        
        auto res = integrate_particle(
            p[0], p[1], p[2], p[3],
            alpha, u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds, thermo_mode, thermo_interp
        );
        
        r[0] = res.first[0];
        r[1] = res.first[1];
        r[2] = res.first[2];
        r[3] = res.first[3];
        if (thermo_results) {
            thermo_results[i] = res.second;
        }
    }
}
