// Enable math constants on Windows
#define _USE_MATH_DEFINES
#include "rk4_integrator.h"
#include <cmath>
#include <algorithm>

// Define M_PI if not available (Windows MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double R_EARTH = 6371000.0; // Earth radius in meters
const double MIN_COS_FACTOR = std::cos(85.0 * M_PI / 180.0); // cos(85 degrees)

RK4Integrator::RK4Integrator(double dt_seconds, double data_interval_hours)
    : dt_seconds_(dt_seconds), data_interval_hours_(data_interval_hours) {}

std::array<double, 3> RK4Integrator::get_velocity(
    double lat, double lon, double pressure,
    double alpha,
    const RegularGrid3DInterpolator& u_curr,
    const RegularGrid3DInterpolator& v_curr, 
    const RegularGrid3DInterpolator& w_curr,
    const RegularGrid3DInterpolator& u_next,
    const RegularGrid3DInterpolator& v_next,
    const RegularGrid3DInterpolator& w_next,
    const std::array<double, 6>& bounds
) const {
    
    // Clip coordinates to bounds
    double clipped_lat = std::clamp(lat, bounds[0], bounds[1]);
    double clipped_lon = std::clamp(lon, bounds[2], bounds[3]);
    double clipped_pressure = std::clamp(pressure, bounds[4], bounds[5]);
    
    // Get velocities from current and next time steps
    double u_c = u_curr.interpolate(clipped_lat, clipped_lon, clipped_pressure);
    double v_c = v_curr.interpolate(clipped_lat, clipped_lon, clipped_pressure);
    double w_c = w_curr.interpolate(clipped_lat, clipped_lon, clipped_pressure);
    
    double u_n = u_next.interpolate(clipped_lat, clipped_lon, clipped_pressure);
    double v_n = v_next.interpolate(clipped_lat, clipped_lon, clipped_pressure);
    double w_n = w_next.interpolate(clipped_lat, clipped_lon, clipped_pressure);
    
    // Temporal interpolation
    double u = (1.0 - alpha) * u_c + alpha * u_n;
    double v = (1.0 - alpha) * v_c + alpha * v_n;
    double w_pa_s = (1.0 - alpha) * w_c + alpha * w_n;
    
    // Check for NaN values
    if (std::isnan(u) || std::isnan(v) || std::isnan(w_pa_s)) {
        return {0.0, 0.0, 0.0};
    }
    
    // Convert to coordinate derivatives
    double lat_rad = lat * M_PI / 180.0;
    double cos_lat = std::cos(lat_rad);
    double lon_denom = R_EARTH * std::max(cos_lat, MIN_COS_FACTOR);
    
    double delta_lat_deg_s = v / R_EARTH * 180.0 / M_PI;
    double delta_lon_deg_s = (lon_denom > 1e-6) ? u / lon_denom * 180.0 / M_PI : 0.0;
    double delta_p_hpa_s = w_pa_s / 100.0; // Convert Pa/s to hPa/s
    
    return {delta_lat_deg_s, delta_lon_deg_s, delta_p_hpa_s};
}

std::array<double, 4> RK4Integrator::integrate_particle(
    double particle_id, double lat, double lon, double pressure,
    double alpha,
    const RegularGrid3DInterpolator& u_curr,
    const RegularGrid3DInterpolator& v_curr,
    const RegularGrid3DInterpolator& w_curr,
    const RegularGrid3DInterpolator& u_next,
    const RegularGrid3DInterpolator& v_next,
    const RegularGrid3DInterpolator& w_next,
    const std::array<double, 6>& bounds
) const {
    
    std::array<double, 3> pos = {lat, lon, pressure};
    
    // RK4 stages
    auto k1 = get_velocity(pos[0], pos[1], pos[2], alpha,
                          u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds);
    
    auto pos_k2 = pos;
    for (int i = 0; i < 3; ++i) {
        pos_k2[i] += 0.5 * dt_seconds_ * k1[i];
    }
    double alpha_k2 = alpha + 0.5 * (dt_seconds_ / 3600.0 / data_interval_hours_);
    auto k2 = get_velocity(pos_k2[0], pos_k2[1], pos_k2[2], alpha_k2,
                          u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds);
    
    auto pos_k3 = pos;
    for (int i = 0; i < 3; ++i) {
        pos_k3[i] += 0.5 * dt_seconds_ * k2[i];
    }
    double alpha_k3 = alpha + 0.5 * (dt_seconds_ / 3600.0 / data_interval_hours_);
    auto k3 = get_velocity(pos_k3[0], pos_k3[1], pos_k3[2], alpha_k3,
                          u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds);
    
    auto pos_k4 = pos;
    for (int i = 0; i < 3; ++i) {
        pos_k4[i] += dt_seconds_ * k3[i];
    }
    double alpha_k4 = alpha + 1.0 * (dt_seconds_ / 3600.0 / data_interval_hours_);
    auto k4 = get_velocity(pos_k4[0], pos_k4[1], pos_k4[2], alpha_k4,
                          u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds);
    
    // Combine RK4 stages
    std::array<double, 3> delta_pos;
    for (int i = 0; i < 3; ++i) {
        delta_pos[i] = (dt_seconds_ / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }
    
    // Update position
    double new_lat = pos[0] + delta_pos[0];
    double new_lon = pos[1] + delta_pos[1];
    double new_pressure = pos[2] + delta_pos[2];
    
    // Apply global physical boundaries
    // Clamp latitude to [-90, 90]
    new_lat = std::clamp(new_lat, -90.0, 90.0);
    
    // Wrap longitude to [-180, 180)
    // This formula correctly handles positive and negative longitudes.
    new_lon = new_lon - std::floor((new_lon + 180.0) / 360.0) * 360.0;
    // Optional: if your convention is (-180, 180] and new_lon becomes exactly -180, adjust to 180.0
    // if (new_lon == -180.0) { new_lon = 180.0; }
    
    // Clamp pressure (already present and correct)
    new_pressure = std::clamp(new_pressure, bounds[4], bounds[5]);
    return {particle_id, new_lat, new_lon, new_pressure};
}

void RK4Integrator::integrate_batch(
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
) const {
    
    size_t n = particles.size();
    results.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        const auto& particle = particles[i];
        results[i] = integrate_particle(
            particle[0], particle[1], particle[2], particle[3],
            alpha, u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds
        );
    }
}