#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <vector>
#include <array>
#include <memory>

class RegularGrid3DInterpolator {
public:
    RegularGrid3DInterpolator(
        const std::vector<double>& lat_coords,
        const std::vector<double>& lon_coords, 
        const std::vector<double>& pressure_coords,
        const std::vector<double>& values,
        double fill_value = 0.0
    );
    
    ~RegularGrid3DInterpolator() = default;
    
    // Interpolate at a single point
    double interpolate(double lat, double lon, double pressure) const;
    
    // Batch interpolation for multiple points
    void interpolate_batch(
        const std::vector<double>& lats,
        const std::vector<double>& lons,
        const std::vector<double>& pressures,
        std::vector<double>& results
    ) const;
    
    // Get coordinate bounds
    std::array<double, 6> get_bounds() const;
    
private:
    std::vector<double> lat_coords_;
    std::vector<double> lon_coords_;
    std::vector<double> pressure_coords_;
    std::vector<double> values_;
    double fill_value_;
    
    size_t nlat_, nlon_, npressure_;
    double lat_min_, lat_max_, lon_min_, lon_max_, pressure_min_, pressure_max_;
    double dlat_, dlon_, dpressure_;
    // Helper functions
    size_t find_index(const std::vector<double>& coords, double value, size_t max_idx) const;
    double trilinear_interpolate(
        size_t i0, size_t j0, size_t k0,
        double t_lat, double t_lon, double t_pressure
    ) const;
    
    inline size_t get_linear_index(size_t i, size_t j, size_t k) const {
        return i * nlon_ * npressure_ + j * npressure_ + k;
    }
};

#endif // INTERPOLATOR_H