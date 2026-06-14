"""
Parameter validation and constraint system for interactive seismic modeling tools.
Provides real-time validation, dependency management, and constraint enforcement.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ParameterType(Enum):
    """Types of parameters for validation."""
    FLOAT = "float"
    INT = "int"
    POSITIVE = "positive"
    RANGE = "range"
    DEPENDENT = "dependent"

@dataclass
class ParameterConstraint:
    """Defines constraints for a parameter."""
    min_value: Union[float, int]
    max_value: Union[float, int]
    step: Union[float, int]
    unit: str
    param_type: ParameterType = ParameterType.FLOAT
    dependencies: Optional[List[str]] = None
    validation_func: Optional[callable] = None
    error_message: Optional[str] = None

class ParameterValidator:
    """
    Advanced parameter validation system with real-time constraint checking,
    dependency management, and geological plausibility validation.
    """
    
    def __init__(self):
        """Initialize the parameter validator with seismic modeling constraints."""
        self.constraints = self._initialize_constraints()
        self.validation_history = []
        self.max_history = 100
    
    def _initialize_constraints(self) -> Dict[str, ParameterConstraint]:
        """Initialize parameter constraints for seismic modeling."""
        return {
            # Wedge model parameters
            'max_thickness': ParameterConstraint(
                min_value=1, max_value=1000, step=1, unit='m',
                param_type=ParameterType.POSITIVE,
                error_message="Thickness must be positive"
            ),
            
            # P-wave velocities (m/s)
            'v1': ParameterConstraint(
                min_value=1000, max_value=8000, step=50, unit='m/s',
                param_type=ParameterType.RANGE,
                dependencies=['v2', 'v3'],
                validation_func=self._validate_velocity_sequence,
                error_message="V1 must be <= V2 <= V3"
            ),
            'v2': ParameterConstraint(
                min_value=1000, max_value=8000, step=50, unit='m/s',
                param_type=ParameterType.RANGE,
                dependencies=['v1', 'v3'],
                validation_func=self._validate_velocity_sequence,
                error_message="V2 must be between V1 and V3"
            ),
            'v3': ParameterConstraint(
                min_value=1000, max_value=8000, step=50, unit='m/s',
                param_type=ParameterType.RANGE,
                dependencies=['v1', 'v2'],
                validation_func=self._validate_velocity_sequence,
                error_message="V3 must be >= V2 >= V1"
            ),
            
            # Densities (g/cc)
            'rho1': ParameterConstraint(
                min_value=1.0, max_value=3.5, step=0.05, unit='g/cc',
                param_type=ParameterType.RANGE,
                dependencies=['rho2', 'rho3'],
                validation_func=self._validate_density_sequence,
                error_message="Rho1 must be <= Rho2 <= Rho3"
            ),
            'rho2': ParameterConstraint(
                min_value=1.0, max_value=3.5, step=0.05, unit='g/cc',
                param_type=ParameterType.RANGE,
                dependencies=['rho1', 'rho3'],
                validation_func=self._validate_density_sequence,
                error_message="Rho2 must be between Rho1 and Rho3"
            ),
            'rho3': ParameterConstraint(
                min_value=1.0, max_value=3.5, step=0.05, unit='g/cc',
                param_type=ParameterType.RANGE,
                dependencies=['rho1', 'rho2'],
                validation_func=self._validate_density_sequence,
                error_message="Rho3 must be >= Rho2 >= Rho1"
            ),
            
            # Wavelet parameters
            'wavelet_freq': ParameterConstraint(
                min_value=5, max_value=200, step=1, unit='Hz',
                param_type=ParameterType.POSITIVE,
                validation_func=self._validate_frequency,
                error_message="Frequency must be positive and reasonable"
            ),
            
            # Display parameters
            'gain': ParameterConstraint(
                min_value=0.1, max_value=5.0, step=0.1, unit='',
                param_type=ParameterType.POSITIVE,
                error_message="Gain must be positive"
            ),
            'plotpadtime': ParameterConstraint(
                min_value=10, max_value=200, step=5, unit='ms',
                param_type=ParameterType.POSITIVE,
                error_message="Plot padding must be positive"
            ),
            
            # Ricker wavelet parameters
            'frequency': ParameterConstraint(
                min_value=5, max_value=200, step=1, unit='Hz',
                param_type=ParameterType.POSITIVE,
                validation_func=self._validate_frequency,
                error_message="Frequency must be positive and reasonable"
            ),
            'time_length': ParameterConstraint(
                min_value=64, max_value=512, step=16, unit='ms',
                param_type=ParameterType.POSITIVE,
                error_message="Time length must be positive"
            ),
            'dt': ParameterConstraint(
                min_value=0.0001, max_value=0.01, step=0.0001, unit='s',
                param_type=ParameterType.POSITIVE,
                validation_func=self._validate_sampling_rate,
                error_message="Sampling interval must be positive and reasonable"
            ),
            
            # AVO parameters
            'angle_min': ParameterConstraint(
                min_value=0, max_value=89, step=1, unit='degrees',
                param_type=ParameterType.RANGE,
                dependencies=['angle_max'],
                validation_func=self._validate_angle_range,
                error_message="Minimum angle must be < maximum angle"
            ),
            'angle_max': ParameterConstraint(
                min_value=1, max_value=90, step=1, unit='degrees',
                param_type=ParameterType.RANGE,
                dependencies=['angle_min'],
                validation_func=self._validate_angle_range,
                error_message="Maximum angle must be > minimum angle"
            ),
            'angle_step': ParameterConstraint(
                min_value=1, max_value=10, step=1, unit='degrees',
                param_type=ParameterType.POSITIVE,
                error_message="Angle step must be positive"
            )
        }
    
    def _validate_velocity_sequence(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """Validate that velocities are in ascending order."""
        v1, v2, v3 = params.get('v1', 0), params.get('v2', 0), params.get('v3', 0)
        
        if v1 <= v2 <= v3:
            return True, ""
        else:
            return False, f"Velocity sequence invalid: V1({v1}) <= V2({v2}) <= V3({v3})"
    
    def _validate_density_sequence(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """Validate that densities are in ascending order."""
        rho1, rho2, rho3 = params.get('rho1', 0), params.get('rho2', 0), params.get('rho3', 0)
        
        if rho1 <= rho2 <= rho3:
            return True, ""
        else:
            return False, f"Density sequence invalid: Rho1({rho1}) <= Rho2({rho2}) <= Rho3({rho3})"
    
    def _validate_frequency(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """Validate frequency is reasonable for seismic applications."""
        freq = params.get('wavelet_freq', params.get('frequency', 0))
        
        if 5 <= freq <= 200:
            return True, ""
        else:
            return False, f"Frequency {freq} Hz is outside reasonable range (5-200 Hz)"
    
    def _validate_sampling_rate(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """Validate sampling rate is appropriate for the frequency."""
        dt = params.get('dt', 0)
        freq = params.get('wavelet_freq', params.get('frequency', 30))
        
        # Nyquist frequency
        nyquist_freq = 1 / (2 * dt)
        
        if nyquist_freq >= 2 * freq:  # At least 2x the frequency
            return True, ""
        else:
            return False, f"Sampling rate too low: dt={dt}s gives Nyquist={nyquist_freq:.1f}Hz < 2*freq={2*freq}Hz"
    
    def _validate_angle_range(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """Validate angle range for AVO analysis."""
        angle_min = params.get('angle_min', 0)
        angle_max = params.get('angle_max', 90)
        
        if angle_min < angle_max:
            return True, ""
        else:
            return False, f"Angle range invalid: min({angle_min}) must be < max({angle_max})"
    
    def validate_parameter(self, param_name: str, value: Union[float, int], 
                          all_params: Optional[Dict[str, Union[float, int]]] = None) -> Tuple[bool, str, Union[float, int]]:
        """
        Validate a single parameter against its constraints.
        
        Args:
            param_name: Name of the parameter
            value: Value to validate
            all_params: All current parameters for dependency validation
            
        Returns:
            Tuple of (is_valid, error_message, corrected_value)
        """
        if param_name not in self.constraints:
            return True, "", value
        
        constraint = self.constraints[param_name]
        corrected_value = value
        
        # Basic range validation
        if value < constraint.min_value:
            corrected_value = constraint.min_value
            return False, f"{param_name} adjusted to minimum {constraint.min_value} {constraint.unit}", corrected_value
        elif value > constraint.max_value:
            corrected_value = constraint.max_value
            return False, f"{param_name} adjusted to maximum {constraint.max_value} {constraint.unit}", corrected_value
        
        # Type validation
        if constraint.param_type == ParameterType.INT:
            corrected_value = int(round(corrected_value))
        elif constraint.param_type == ParameterType.POSITIVE and corrected_value <= 0:
            corrected_value = constraint.min_value
            return False, f"{param_name} must be positive", corrected_value
        
        # Dependency validation
        if constraint.validation_func and all_params:
            all_params_copy = all_params.copy()
            all_params_copy[param_name] = corrected_value
            is_valid, error_msg = constraint.validation_func(all_params_copy)
            if not is_valid:
                return False, error_msg, corrected_value
        
        return True, "", corrected_value
    
    def validate_parameters(self, parameters: Dict[str, Union[float, int]]) -> Tuple[bool, List[str], Dict[str, Union[float, int]]]:
        """
        Validate a set of parameters with dependency checking.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Tuple of (all_valid, error_messages, corrected_parameters)
        """
        corrected = parameters.copy()
        errors = []
        
        # First pass: validate individual parameters
        for param_name, value in parameters.items():
            is_valid, error_msg, corrected_value = self.validate_parameter(
                param_name, value, corrected
            )
            corrected[param_name] = corrected_value
            if not is_valid:
                errors.append(error_msg)
        
        # Second pass: validate dependencies
        for param_name, constraint in self.constraints.items():
            if constraint.validation_func and param_name in corrected:
                is_valid, error_msg = constraint.validation_func(corrected)
                if not is_valid:
                    errors.append(error_msg)
        
        # Add to validation history
        self._add_to_history(corrected, errors)
        
        return len(errors) == 0, errors, corrected
    
    def _add_to_history(self, parameters: Dict[str, Union[float, int]], errors: List[str]):
        """Add validation result to history."""
        self.validation_history.append({
            'parameters': parameters.copy(),
            'errors': errors.copy(),
            'timestamp': np.datetime64('now')
        })
        
        if len(self.validation_history) > self.max_history:
            self.validation_history.pop(0)
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()
    
    def suggest_corrections(self, parameters: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """
        Suggest corrections for invalid parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Dictionary with suggested corrections and explanations
        """
        suggestions = {}
        
        for param_name, value in parameters.items():
            if param_name not in self.constraints:
                continue
            
            constraint = self.constraints[param_name]
            
            # Suggest based on geological plausibility
            if param_name.startswith('v') and isinstance(value, (int, float)):
                if value < 1500:
                    suggestions[param_name] = {
                        'suggested_value': 2000,
                        'reason': 'Velocity too low for consolidated rock',
                        'geological_context': 'Typical rock velocities start around 2000 m/s'
                    }
                elif value > 6000:
                    suggestions[param_name] = {
                        'suggested_value': 4000,
                        'reason': 'Velocity very high, check units',
                        'geological_context': 'Most sedimentary rocks have velocities < 5000 m/s'
                    }
            
            elif param_name.startswith('rho') and isinstance(value, (int, float)):
                if value < 1.5:
                    suggestions[param_name] = {
                        'suggested_value': 2.0,
                        'reason': 'Density too low for rock',
                        'geological_context': 'Most rocks have density > 2.0 g/cc'
                    }
                elif value > 3.0:
                    suggestions[param_name] = {
                        'suggested_value': 2.5,
                        'reason': 'Density very high, check units',
                        'geological_context': 'Most sedimentary rocks have density < 3.0 g/cc'
                    }
        
        return suggestions
    
    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a parameter."""
        if param_name not in self.constraints:
            return None
        
        constraint = self.constraints[param_name]
        return {
            'name': param_name,
            'min_value': constraint.min_value,
            'max_value': constraint.max_value,
            'step': constraint.step,
            'unit': constraint.unit,
            'param_type': constraint.param_type.value,
            'dependencies': constraint.dependencies or [],
            'error_message': constraint.error_message
        }
    
    def get_all_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all parameters."""
        return {name: self.get_parameter_info(name) for name in self.constraints.keys()}

# Global validator instance
_global_validator = None

def get_global_validator() -> ParameterValidator:
    """Get the global parameter validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = ParameterValidator()
    return _global_validator

def validate_seismic_parameters(parameters: Dict[str, Union[float, int]]) -> Tuple[bool, List[str], Dict[str, Union[float, int]]]:
    """
    Convenience function to validate seismic parameters using the global validator.
    
    Args:
        parameters: Dictionary of parameter values
        
    Returns:
        Tuple of (all_valid, error_messages, corrected_parameters)
    """
    validator = get_global_validator()
    return validator.validate_parameters(parameters)

def suggest_parameter_corrections(parameters: Dict[str, Union[float, int]]) -> Dict[str, Any]:
    """
    Convenience function to suggest parameter corrections.

    Args:
        parameters: Dictionary of parameter values

    Returns:
        Dictionary with suggested corrections
    """
    validator = get_global_validator()
    return validator.suggest_corrections(parameters)


# --- Per-tool validators referenced by core.tool_registry ---
from typing import Tuple as _Tuple, Dict as _Dict, Any as _Any


def validate_make_ricker(params: _Dict[str, _Any]) -> _Tuple[bool, str]:
    freq = params.get("frequency")
    if not freq or freq <= 0 or freq > 1000:
        return False, "Frequency must be between 0 and 1000 Hz"
    dt = params.get("dt", 0.001)
    if dt <= 0 or dt > 0.1:
        return False, "Sampling interval (dt) must be between 0 and 0.1 seconds"
    return True, ""


def validate_wedge_model(params: _Dict[str, _Any]) -> _Tuple[bool, str]:
    thickness = params.get("max_thickness")
    if not thickness or thickness <= 0:
        return False, "Maximum thickness must be positive"
    for i in range(1, 4):
        v = params.get(f"v{i}")
        if not v or v <= 0:
            return False, f"Velocity v{i} must be positive"
        if v > 6500 or v < 1500:
            return False, f"Invalid v{i}: must be between 1500 and 6500 m/s"
    for i in range(1, 4):
        rho = params.get(f"rho{i}")
        if not rho or rho <= 0:
            return False, f"Density rho{i} must be positive"
    return True, ""


def validate_avo(params: _Dict[str, _Any]) -> _Tuple[bool, str]:
    for p in ["vp1", "vs1", "rho1", "vp2", "vs2", "rho2", "angles"]:
        if p not in params:
            return False, f"Missing required parameter: {p}"
    return True, ""

