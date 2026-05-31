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
const double KAPPA = 0.286;
const double P0_HPA = 1000.0;

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
    
    // Clip coordinates to bounds (respecting periodicity)
    double clipped_lat = std::clamp(lat, bounds[0], bounds[1]);
    double clipped_lon = lon;
    if (!u_curr.is_periodic_lon()) {
        clipped_lon = std::clamp(lon, bounds[2], bounds[3]);
    }
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

// Helper for scalar interpolation
static double interpolate_scalar(
    double lat, double lon, double pressure, double alpha,
    const RegularGrid3DInterpolator* curr,
    const RegularGrid3DInterpolator* next,
    const std::array<double, 6>& bounds) {
    if (!curr || !next) return 0.0;
    double c_lat = std::clamp(lat, bounds[0], bounds[1]);
    double c_lon = lon;
    if (!curr->is_periodic_lon()) {
        c_lon = std::clamp(lon, bounds[2], bounds[3]);
    }
    double c_p = std::clamp(pressure, bounds[4], bounds[5]);
    double val_curr = curr->interpolate(c_lat, c_lon, c_p);
    double val_next = next->interpolate(c_lat, c_lon, c_p);
    return (1.0 - alpha) * val_curr + alpha * val_next;
}

std::pair<std::array<double, 4>, ThermoState> RK4Integrator::integrate_particle(
    double particle_id, double lat, double lon, double pressure,
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
    
    std::array<double, 3> pos = {lat, lon, pressure};
    
    // Lambda to sanitize/wrap coordinates at each RK4 sub-step
    auto sanitize_coords = [&](std::array<double, 3>& p) {
        // Polar boundary crossing wrapping
        if (p[0] > 90.0) {
            p[0] = 180.0 - p[0];
            p[1] += 180.0;
        } else if (p[0] < -90.0) {
            p[0] = -180.0 - p[0];
            p[1] += 180.0;
        }
        // Longitude wrapping to standard [-180, 180) range (or wrapped inside interpolator anyway)
        p[1] = p[1] - std::floor((p[1] + 180.0) / 360.0) * 360.0;
        // Pressure clamping
        p[2] = std::clamp(p[2], bounds[4], bounds[5]);
    };
    
    // Sanitize initial coordinate just in case
    sanitize_coords(pos);
    
    // RK4 stages
    auto k1 = get_velocity(pos[0], pos[1], pos[2], alpha,
                          u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds);
    
    auto pos_k2 = pos;
    for (int i = 0; i < 3; ++i) {
        pos_k2[i] += 0.5 * dt_seconds_ * k1[i];
    }
    sanitize_coords(pos_k2);
    double alpha_k2 = alpha + 0.5 * (dt_seconds_ / 3600.0 / data_interval_hours_);
    auto k2 = get_velocity(pos_k2[0], pos_k2[1], pos_k2[2], alpha_k2,
                          u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds);
    
    auto pos_k3 = pos;
    for (int i = 0; i < 3; ++i) {
        pos_k3[i] += 0.5 * dt_seconds_ * k2[i];
    }
    sanitize_coords(pos_k3);
    double alpha_k3 = alpha + 0.5 * (dt_seconds_ / 3600.0 / data_interval_hours_);
    auto k3 = get_velocity(pos_k3[0], pos_k3[1], pos_k3[2], alpha_k3,
                          u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds);
    
    auto pos_k4 = pos;
    for (int i = 0; i < 3; ++i) {
        pos_k4[i] += dt_seconds_ * k3[i];
    }
    sanitize_coords(pos_k4);
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
    
    std::array<double, 3> final_pos = {new_lat, new_lon, new_pressure};
    sanitize_coords(final_pos);
    
    new_lat = final_pos[0];
    new_lon = final_pos[1];
    new_pressure = final_pos[2];
    
    std::array<double, 4> new_particle = {particle_id, new_lat, new_lon, new_pressure};
    ThermoState delta_ts;
    
    // Thermodynamic Calculations
    if (thermo_mode != ThermoMode::NONE) {
        if (thermo_mode == ThermoMode::SIMPLE_SUBTRACTION) {
            double t_start = interpolate_scalar(lat, lon, pressure, alpha, thermo_interp.t_curr, thermo_interp.t_next, bounds);
            double t_end = interpolate_scalar(new_lat, new_lon, new_pressure, alpha_k4, thermo_interp.t_curr, thermo_interp.t_next, bounds);
            delta_ts.values[0] = t_end - t_start;
        } 
        else if (thermo_mode == ThermoMode::POTENTIAL_TEMPERATURE) {
            double t_start = interpolate_scalar(lat, lon, pressure, alpha, thermo_interp.t_curr, thermo_interp.t_next, bounds);
            double t_end = interpolate_scalar(new_lat, new_lon, new_pressure, alpha_k4, thermo_interp.t_curr, thermo_interp.t_next, bounds);
            double theta_start = t_start * std::pow(P0_HPA / pressure, KAPPA);
            double theta_end = t_end * std::pow(P0_HPA / new_pressure, KAPPA);
            delta_ts.values[0] = theta_end - theta_start; // delta_theta
            delta_ts.values[1] = 0.0; // delta_DSE (needs z, omitted for now)
        }
        else if (thermo_mode == ThermoMode::FULL_DECOMPOSITION) {
            // Helper lambda to evaluate integrands at a point (lat, lon, p, alpha)
            auto evaluate_terms = [&](double l_lat, double l_lon, double l_p, double l_alpha) -> std::array<double, 3> {
                double u_c = u_curr.interpolate(l_lat, l_lon, l_p);
                double v_c = v_curr.interpolate(l_lat, l_lon, l_p);
                double w_c = w_curr.interpolate(l_lat, l_lon, l_p);
                double u_n = u_next.interpolate(l_lat, l_lon, l_p);
                double v_n = v_next.interpolate(l_lat, l_lon, l_p);
                double w_n = w_next.interpolate(l_lat, l_lon, l_p);
                double u = (1.0 - l_alpha) * u_c + l_alpha * u_n;
                double v = (1.0 - l_alpha) * v_c + l_alpha * v_n;
                double w_pa_s = (1.0 - l_alpha) * w_c + l_alpha * w_n; // omega in Pa/s
                
                double dt_mean_dt = interpolate_scalar(l_lat, l_lon, l_p, l_alpha, thermo_interp.dt_mean_dt_curr, thermo_interp.dt_mean_dt_next, bounds);
                double grad_lat = interpolate_scalar(l_lat, l_lon, l_p, l_alpha, thermo_interp.grad_t_lat_curr, thermo_interp.grad_t_lat_next, bounds);
                double grad_lon = interpolate_scalar(l_lat, l_lon, l_p, l_alpha, thermo_interp.grad_t_lon_curr, thermo_interp.grad_t_lon_next, bounds);
                double grad_p = interpolate_scalar(l_lat, l_lon, l_p, l_alpha, thermo_interp.grad_t_p_curr, thermo_interp.grad_t_p_next, bounds);
                double t_mean = interpolate_scalar(l_lat, l_lon, l_p, l_alpha, thermo_interp.t_mean_curr, thermo_interp.t_mean_next, bounds);
                
                double term1_seasonality = -dt_mean_dt;
                double term2_advective = -(u * grad_lon + v * grad_lat);
                
                // Adiabatic term: (kappa * t_mean / p_pa - grad_p) * omega
                // Note: l_p is in hPa, so p_pa = l_p * 100
                double p_pa = l_p * 100.0;
                double term3_adiabatic = ((KAPPA * t_mean / p_pa) - grad_p) * w_pa_s;
                
                return {term1_seasonality, term2_advective, term3_adiabatic};
            };
            
            auto start_terms = evaluate_terms(lat, lon, pressure, alpha);
            auto end_terms = evaluate_terms(new_lat, new_lon, new_pressure, alpha_k4);
            
            // Trapezoidal rule integration over dt_seconds_
            delta_ts.values[0] = 0.5 * dt_seconds_ * (start_terms[0] + end_terms[0]); // Seasonality
            delta_ts.values[1] = 0.5 * dt_seconds_ * (start_terms[1] + end_terms[1]); // Advective
            delta_ts.values[2] = 0.5 * dt_seconds_ * (start_terms[2] + end_terms[2]); // Adiabatic
            
            // Diabatic term approx: (p_avg / P0)^kappa * (theta_end - theta_start)
            double t_start = interpolate_scalar(lat, lon, pressure, alpha, thermo_interp.t_curr, thermo_interp.t_next, bounds);
            double t_end = interpolate_scalar(new_lat, new_lon, new_pressure, alpha_k4, thermo_interp.t_curr, thermo_interp.t_next, bounds);
            double theta_start = t_start * std::pow(P0_HPA / pressure, KAPPA);
            double theta_end = t_end * std::pow(P0_HPA / new_pressure, KAPPA);
            double p_avg = 0.5 * (pressure + new_pressure);
            
            delta_ts.values[3] = std::pow(p_avg / P0_HPA, KAPPA) * (theta_end - theta_start);
        }
    }
    
    return {new_particle, delta_ts};
}

void RK4Integrator::integrate_batch(
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
    ThermoMode thermo_mode,
    const ThermoInterpolators& thermo_interp
) const {
    size_t n = particles.size();
    results.resize(n);
    thermo_results.resize(n);
    
    integrate_batch(
        reinterpret_cast<const double*>(particles.data()),
        reinterpret_cast<double*>(results.data()),
        thermo_results.data(),
        n,
        alpha, u_curr, v_curr, w_curr, u_next, v_next, w_next, bounds, thermo_mode, thermo_interp
    );
}

void RK4Integrator::integrate_batch(
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
    for (size_t i = 0; i < count; ++i) {
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