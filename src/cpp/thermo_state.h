#ifndef THERMO_STATE_H
#define THERMO_STATE_H

#include <string>

// Enum for thermodynamic mode
enum class ThermoMode {
    NONE = -1,
    SIMPLE_SUBTRACTION = 0,
    POTENTIAL_TEMPERATURE = 1,
    FULL_DECOMPOSITION = 2
};

inline ThermoMode string_to_thermo_mode(const std::string& mode_str) {
    if (mode_str == "SIMPLE_SUBTRACTION") return ThermoMode::SIMPLE_SUBTRACTION;
    if (mode_str == "POTENTIAL_TEMPERATURE") return ThermoMode::POTENTIAL_TEMPERATURE;
    if (mode_str == "FULL_DECOMPOSITION") return ThermoMode::FULL_DECOMPOSITION;
    return ThermoMode::NONE; // Default or disabled
}

// Struct to hold thermodynamic state variables
// To keep it simple for numpy integration, we will define the max number of fields we need.
// For Simple Subtraction: delta_T (1 field)
// For Potential Temperature: delta_theta, delta_DSE (2 fields)
// For Full Decomposition: seasonality_term, advective_term, adiabatic_term, diabatic_term (4 fields)
// We will allocate a fixed size array of 4 doubles to cover the maximum needed variables.
struct ThermoState {
    double values[4] = {0.0, 0.0, 0.0, 0.0};
};

class RegularGrid3DInterpolator; // Forward declaration

// Struct to hold pointers to interpolators for thermodynamic fields
// Pointers can be null if the field is not provided or not needed for the chosen thermo_mode
struct ThermoInterpolators {
    const RegularGrid3DInterpolator* t_curr = nullptr;
    const RegularGrid3DInterpolator* t_next = nullptr;
    
    // Additional fields for Method 2
    const RegularGrid3DInterpolator* t_mean_curr = nullptr;
    const RegularGrid3DInterpolator* t_mean_next = nullptr;
    const RegularGrid3DInterpolator* grad_t_lat_curr = nullptr;
    const RegularGrid3DInterpolator* grad_t_lat_next = nullptr;
    const RegularGrid3DInterpolator* grad_t_lon_curr = nullptr;
    const RegularGrid3DInterpolator* grad_t_lon_next = nullptr;
    const RegularGrid3DInterpolator* grad_t_p_curr = nullptr;
    const RegularGrid3DInterpolator* grad_t_p_next = nullptr;
    const RegularGrid3DInterpolator* dt_mean_dt_curr = nullptr;
    const RegularGrid3DInterpolator* dt_mean_dt_next = nullptr;
};

#endif // THERMO_STATE_H
