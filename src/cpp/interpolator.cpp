#include "interpolator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

RegularGrid3DInterpolator::RegularGrid3DInterpolator(
    const std::vector<double>& lat_coords,
    const std::vector<double>& lon_coords,
    const std::vector<double>& pressure_coords,
    const std::vector<double>& values,
    double fill_value
) : is_periodic_lon_(false), lat_coords_(lat_coords), lon_coords_(lon_coords), pressure_coords_(pressure_coords),
    values_(values), fill_value_(fill_value) {
    
    nlat_ = lat_coords_.size();
    nlon_ = lon_coords_.size();
    npressure_ = pressure_coords_.size();
    
    if (values_.size() != nlat_ * nlon_ * npressure_) {
        throw std::invalid_argument("Values array size doesn't match coordinate dimensions");
    }
    
    if (nlat_ < 2 || nlon_ < 2 || npressure_ < 2) {
        throw std::invalid_argument("Need at least 2 points in each dimension");
    }
    
    // Store bounds and spacing
    lat_min_ = lat_coords_[0];
    lat_max_ = lat_coords_[nlat_ - 1];
    lon_min_ = lon_coords_[0];
    lon_max_ = lon_coords_[nlon_ - 1];
    pressure_min_ = pressure_coords_[0];
    pressure_max_ = pressure_coords_[npressure_ - 1];
    
    dlat_ = (lat_max_ - lat_min_) / (nlat_ - 1);
    dlon_ = (lon_max_ - lon_min_) / (nlon_ - 1);
    dpressure_ = (pressure_max_ - pressure_min_) / (npressure_ - 1);
    
    // Auto-detect if longitude grid is periodic (global simulation)
    is_periodic_lon_ = ((lon_max_ - lon_min_ + dlon_) >= 360.0 - 1e-3);
}

size_t RegularGrid3DInterpolator::find_index(
    const std::vector<double>& coords, double value, size_t max_idx
) const { // max_idx is the index of the last element in coords (i.e., coords.size() - 1)
    // This function assumes 'value' is already within the bounds [coords[0], coords[max_idx]]
    // due to checks in the main interpolate() method.

    // If value is at the upper boundary, the cell is [max_idx-1, max_idx].
    // The lower index i0 should be max_idx-1.
    // This also correctly handles the case where coords might have only 2 points (max_idx = 1),
    // making i0 = 0.
    if (value == coords[max_idx]) {
        return max_idx - 1;
    }

    // For coords[0] <= value < coords[max_idx]
    // Find i0 such that coords[i0] <= value < coords[i0+1].
    // std::upper_bound returns an iterator to the first element in coords strictly greater than 'value'.
    // The range for upper_bound includes all elements of coords.
    auto it = std::upper_bound(coords.begin(), coords.begin() + max_idx + 1, value);

    // idx_after_value will be the index of the element *it points to.
    // Since coords[0] <= value < coords[max_idx], idx_after_value will be > 0.
    // The lower grid index i0 is then (idx_after_value - 1).
    return std::distance(coords.begin(), it) - 1;
}

double RegularGrid3DInterpolator::trilinear_interpolate(
    size_t i0, size_t j0, size_t k0,
    size_t j1,
    double t_lat, double t_lon, double t_pressure
) const {
    
    size_t i1 = std::min(i0 + 1, nlat_ - 1);
    size_t k1 = std::min(k0 + 1, npressure_ - 1);
    
    // Get the 8 corner values
    double c000 = values_[get_linear_index(i0, j0, k0)];
    double c001 = values_[get_linear_index(i0, j0, k1)];
    double c010 = values_[get_linear_index(i0, j1, k0)];
    double c011 = values_[get_linear_index(i0, j1, k1)];
    double c100 = values_[get_linear_index(i1, j0, k0)];
    double c101 = values_[get_linear_index(i1, j0, k1)];
    double c110 = values_[get_linear_index(i1, j1, k0)];
    double c111 = values_[get_linear_index(i1, j1, k1)];
    
    // Check for NaN values
    if (std::isnan(c000) || std::isnan(c001) || std::isnan(c010) || std::isnan(c011) ||
        std::isnan(c100) || std::isnan(c101) || std::isnan(c110) || std::isnan(c111)) {
        return fill_value_;
    }
    
    // Trilinear interpolation
    double c00 = c000 * (1 - t_pressure) + c001 * t_pressure;
    double c01 = c010 * (1 - t_pressure) + c011 * t_pressure;
    double c10 = c100 * (1 - t_pressure) + c101 * t_pressure;
    double c11 = c110 * (1 - t_pressure) + c111 * t_pressure;
    
    double c0 = c00 * (1 - t_lon) + c01 * t_lon;
    double c1 = c10 * (1 - t_lon) + c11 * t_lon;
    
    return c0 * (1 - t_lat) + c1 * t_lat;
}

double RegularGrid3DInterpolator::interpolate(double lat, double lon, double pressure) const {
    // Check lat/pressure bounds first
    if (lat < lat_min_ || lat > lat_max_ ||
        pressure < pressure_min_ || pressure > pressure_max_) {
        return fill_value_;
    }
    
    double query_lon = lon;
    if (is_periodic_lon_) {
        // Wrap query_lon to range [lon_min_, lon_min_ + 360.0)
        query_lon = lon - std::floor((lon - lon_min_) / 360.0) * 360.0;
    } else {
        // Strict bounds check on longitude for regional simulation
        if (lon < lon_min_ || lon > lon_max_) {
            return fill_value_;
        }
    }
    
    // Find indices
    size_t i0 = find_index(lat_coords_, lat, nlat_ - 1);
    size_t k0 = find_index(pressure_coords_, pressure, npressure_ - 1);
    
    size_t j0 = 0;
    size_t j1 = 0;
    double t_lon = 0.0;
    
    if (is_periodic_lon_ && query_lon >= lon_max_) {
        // Periodic wrap-around grid cell
        j0 = nlon_ - 1;
        j1 = 0;
        t_lon = (query_lon - lon_max_) / dlon_;
    } else {
        j0 = find_index(lon_coords_, query_lon, nlon_ - 1);
        j1 = std::min(j0 + 1, nlon_ - 1);
        if (j0 < nlon_ - 1) {
            t_lon = (query_lon - lon_coords_[j0]) / (lon_coords_[j1] - lon_coords_[j0]);
        }
    }
    
    // Calculate other interpolation weights
    double t_lat = 0.0, t_pressure = 0.0;
    
    if (i0 < nlat_ - 1) {
        t_lat = (lat - lat_coords_[i0]) / (lat_coords_[i0 + 1] - lat_coords_[i0]);
    }
    if (k0 < npressure_ - 1) {
        t_pressure = (pressure - pressure_coords_[k0]) / (pressure_coords_[k0 + 1] - pressure_coords_[k0]);
    }
    
    return trilinear_interpolate(i0, j0, k0, j1, t_lat, t_lon, t_pressure);
}

void RegularGrid3DInterpolator::interpolate_batch(
    const std::vector<double>& lats,
    const std::vector<double>& lons,
    const std::vector<double>& pressures,
    std::vector<double>& results
) const {
    size_t n = lats.size();
    results.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        results[i] = interpolate(lats[i], lons[i], pressures[i]);
    }
}

std::array<double, 6> RegularGrid3DInterpolator::get_bounds() const {
    return {lat_min_, lat_max_, lon_min_, lon_max_, pressure_min_, pressure_max_};
}