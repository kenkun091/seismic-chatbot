"""
Parameter linking and dependency management system for interactive seismic modeling.
Allows parameters to be connected and updated together with real-time synchronization.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

class LinkType(Enum):
    """Types of parameter links."""
    EQUAL = "equal"  # Parameters must be equal
    PROPORTIONAL = "proportional"  # One parameter is proportional to another
    INVERSE = "inverse"  # One parameter is inversely proportional to another
    SEQUENCE = "sequence"  # Parameters must be in ascending/descending order
    CALCULATED = "calculated"  # One parameter is calculated from others
    RANGE = "range"  # One parameter must be within a range of another

@dataclass
class ParameterLink:
    """Defines a link between parameters."""
    source_param: str
    target_param: str
    link_type: LinkType
    factor: float = 1.0
    offset: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enabled: bool = True
    description: str = ""

class ParameterLinkManager:
    """
    Manages parameter links and dependencies for real-time parameter synchronization.
    """
    
    def __init__(self):
        """Initialize the parameter link manager."""
        self.links: List[ParameterLink] = []
        self.link_groups: Dict[str, List[str]] = {}  # Groups of linked parameters
        self.update_callbacks: Dict[str, List[Callable]] = {}  # Callbacks for parameter updates
        self.parameter_values: Dict[str, float] = {}
        self.update_in_progress = False
    
    def add_link(self, link: ParameterLink) -> bool:
        """
        Add a parameter link.
        
        Args:
            link: ParameterLink object defining the relationship
            
        Returns:
            True if link was added successfully
        """
        # Check for circular dependencies
        if self._would_create_circular_dependency(link):
            return False
        
        self.links.append(link)
        self._update_link_groups()
        return True
    
    def remove_link(self, source_param: str, target_param: str) -> bool:
        """
        Remove a parameter link.
        
        Args:
            source_param: Source parameter name
            target_param: Target parameter name
            
        Returns:
            True if link was removed
        """
        for i, link in enumerate(self.links):
            if link.source_param == source_param and link.target_param == target_param:
                del self.links[i]
                self._update_link_groups()
                return True
        return False
    
    def _would_create_circular_dependency(self, new_link: ParameterLink) -> bool:
        """Check if adding a link would create a circular dependency."""
        # Simple check: if target_param is already a source for source_param
        for link in self.links:
            if (link.source_param == new_link.target_param and 
                link.target_param == new_link.source_param):
                return True
        return False
    
    def _update_link_groups(self):
        """Update parameter groups based on current links."""
        self.link_groups = {}
        
        for link in self.links:
            if not link.enabled:
                continue
            
            # Add to groups
            if link.source_param not in self.link_groups:
                self.link_groups[link.source_param] = []
            if link.target_param not in self.link_groups:
                self.link_groups[link.target_param] = []
            
            if link.target_param not in self.link_groups[link.source_param]:
                self.link_groups[link.source_param].append(link.target_param)
            if link.source_param not in self.link_groups[link.target_param]:
                self.link_groups[link.target_param].append(link.source_param)
    
    def set_parameter(self, param_name: str, value: float, 
                     trigger_updates: bool = True) -> Dict[str, float]:
        """
        Set a parameter value and update all linked parameters.
        
        Args:
            param_name: Name of the parameter to set
            value: New value
            trigger_updates: Whether to trigger updates to linked parameters
            
        Returns:
            Dictionary of all updated parameter values
        """
        if self.update_in_progress:
            return self.parameter_values.copy()
        
        self.update_in_progress = True
        
        try:
            self.parameter_values[param_name] = value
            updated_params = {param_name: value}
            
            if trigger_updates:
                # Find all parameters linked to this one
                linked_params = self._get_linked_parameters(param_name)
                
                for linked_param in linked_params:
                    new_value = self._calculate_linked_value(param_name, linked_param)
                    if new_value is not None:
                        self.parameter_values[linked_param] = new_value
                        updated_params[linked_param] = new_value
                
                # Trigger callbacks
                self._trigger_callbacks(param_name, updated_params)
            
            return updated_params
            
        finally:
            self.update_in_progress = False
    
    def _get_linked_parameters(self, param_name: str) -> List[str]:
        """Get all parameters linked to the given parameter."""
        linked = set()
        
        for link in self.links:
            if not link.enabled:
                continue
            
            if link.source_param == param_name:
                linked.add(link.target_param)
            elif link.target_param == param_name:
                linked.add(link.source_param)
        
        return list(linked)
    
    def _calculate_linked_value(self, source_param: str, target_param: str) -> Optional[float]:
        """Calculate the value for a linked parameter."""
        source_value = self.parameter_values.get(source_param)
        if source_value is None:
            return None
        
        # Find the link between these parameters
        link = None
        for l in self.links:
            if ((l.source_param == source_param and l.target_param == target_param) or
                (l.source_param == target_param and l.target_param == source_param)):
                link = l
                break
        
        if not link or not link.enabled:
            return None
        
        # Calculate based on link type
        if link.link_type == LinkType.EQUAL:
            return source_value
        
        elif link.link_type == LinkType.PROPIONAL:
            if link.source_param == source_param:
                return source_value * link.factor + link.offset
            else:
                return (source_value - link.offset) / link.factor
        
        elif link.link_type == LinkType.INVERSE:
            if link.source_param == source_param:
                return link.factor / source_value + link.offset
            else:
                return link.factor / (source_value - link.offset)
        
        elif link.link_type == LinkType.SEQUENCE:
            # For sequence, we need to check other parameters in the sequence
            return self._calculate_sequence_value(source_param, target_param, source_value)
        
        elif link.link_type == LinkType.CALCULATED:
            return self._calculate_custom_value(source_param, target_param, source_value, link)
        
        elif link.link_type == LinkType.RANGE:
            return self._calculate_range_value(source_param, target_param, source_value, link)
        
        return None
    
    def _calculate_sequence_value(self, source_param: str, target_param: str, 
                                source_value: float) -> Optional[float]:
        """Calculate value for sequence-type links."""
        # This is a simplified implementation
        # In practice, you'd need to know the sequence order
        return source_value * 1.1  # Example: 10% increase
    
    def _calculate_custom_value(self, source_param: str, target_param: str, 
                              source_value: float, link: ParameterLink) -> Optional[float]:
        """Calculate value for custom calculated links."""
        # This would be implemented based on specific calculation needs
        return source_value * link.factor + link.offset
    
    def _calculate_range_value(self, source_param: str, target_param: str, 
                             source_value: float, link: ParameterLink) -> Optional[float]:
        """Calculate value for range-type links."""
        if link.min_value is not None and link.max_value is not None:
            # Scale source value to target range
            return link.min_value + (source_value - link.min_value) * link.factor
        return source_value
    
    def _trigger_callbacks(self, changed_param: str, updated_params: Dict[str, float]):
        """Trigger callbacks for parameter updates."""
        if changed_param in self.update_callbacks:
            for callback in self.update_callbacks[changed_param]:
                try:
                    callback(updated_params)
                except Exception as e:
                    print(f"Error in parameter update callback: {e}")
    
    def add_update_callback(self, param_name: str, callback: Callable[[Dict[str, float]], None]):
        """Add a callback for parameter updates."""
        if param_name not in self.update_callbacks:
            self.update_callbacks[param_name] = []
        self.update_callbacks[param_name].append(callback)
    
    def remove_update_callback(self, param_name: str, callback: Callable[[Dict[str, float]], None]):
        """Remove a parameter update callback."""
        if param_name in self.update_callbacks:
            try:
                self.update_callbacks[param_name].remove(callback)
            except ValueError:
                pass
    
    def get_parameter_value(self, param_name: str) -> Optional[float]:
        """Get the current value of a parameter."""
        return self.parameter_values.get(param_name)
    
    def get_all_parameter_values(self) -> Dict[str, float]:
        """Get all current parameter values."""
        return self.parameter_values.copy()
    
    def set_all_parameters(self, parameters: Dict[str, float]):
        """Set multiple parameters at once without triggering updates."""
        self.parameter_values.update(parameters)
    
    def get_linked_parameters(self, param_name: str) -> List[str]:
        """Get all parameters linked to the given parameter."""
        return self._get_linked_parameters(param_name)
    
    def get_link_info(self, param_name: str) -> List[Dict[str, Any]]:
        """Get information about all links involving a parameter."""
        link_info = []
        
        for link in self.links:
            if link.source_param == param_name or link.target_param == param_name:
                link_info.append({
                    'source_param': link.source_param,
                    'target_param': link.target_param,
                    'link_type': link.link_type.value,
                    'factor': link.factor,
                    'offset': link.offset,
                    'enabled': link.enabled,
                    'description': link.description
                })
        
        return link_info
    
    def enable_link(self, source_param: str, target_param: str, enabled: bool = True):
        """Enable or disable a parameter link."""
        for link in self.links:
            if (link.source_param == source_param and 
                link.target_param == target_param):
                link.enabled = enabled
                self._update_link_groups()
                break
    
    def clear_all_links(self):
        """Clear all parameter links."""
        self.links.clear()
        self.link_groups.clear()
        self.update_callbacks.clear()

# Predefined link configurations for common seismic modeling scenarios
class SeismicLinkPresets:
    """Predefined parameter link configurations for seismic modeling."""
    
    @staticmethod
    def create_velocity_density_links(link_manager: ParameterLinkManager) -> List[ParameterLink]:
        """Create links between velocity and density based on empirical relationships."""
        links = []
        
        # Gardner's relationship: Vp = a * ρ^b (simplified)
        # This is a rough approximation
        links.append(ParameterLink(
            source_param='v1',
            target_param='rho1',
            link_type=LinkType.CALCULATED,
            factor=0.31,  # Gardner's constant
            offset=0.23,
            description="Gardner's relationship: Vp = 0.31 * ρ^0.25"
        ))
        
        links.append(ParameterLink(
            source_param='v2',
            target_param='rho2',
            link_type=LinkType.CALCULATED,
            factor=0.31,
            offset=0.23,
            description="Gardner's relationship: Vp = 0.31 * ρ^0.25"
        ))
        
        links.append(ParameterLink(
            source_param='v3',
            target_param='rho3',
            link_type=LinkType.CALCULATED,
            factor=0.31,
            offset=0.23,
            description="Gardner's relationship: Vp = 0.31 * ρ^0.25"
        ))
        
        return links
    
    @staticmethod
    def create_velocity_sequence_links(link_manager: ParameterLinkManager) -> List[ParameterLink]:
        """Create links to maintain velocity sequence (V1 <= V2 <= V3)."""
        links = []
        
        # V2 should be between V1 and V3
        links.append(ParameterLink(
            source_param='v1',
            target_param='v2',
            link_type=LinkType.SEQUENCE,
            factor=1.2,  # V2 = 1.2 * V1
            description="V2 proportional to V1"
        ))
        
        links.append(ParameterLink(
            source_param='v2',
            target_param='v3',
            link_type=LinkType.SEQUENCE,
            factor=1.2,  # V3 = 1.2 * V2
            description="V3 proportional to V2"
        ))
        
        return links
    
    @staticmethod
    def create_density_sequence_links(link_manager: ParameterLinkManager) -> List[ParameterLink]:
        """Create links to maintain density sequence (Rho1 <= Rho2 <= Rho3)."""
        links = []
        
        # Rho2 should be between Rho1 and Rho3
        links.append(ParameterLink(
            source_param='rho1',
            target_param='rho2',
            link_type=LinkType.SEQUENCE,
            factor=1.1,  # Rho2 = 1.1 * Rho1
            description="Rho2 proportional to Rho1"
        ))
        
        links.append(ParameterLink(
            source_param='rho2',
            target_param='rho3',
            link_type=LinkType.SEQUENCE,
            factor=1.1,  # Rho3 = 1.1 * Rho2
            description="Rho3 proportional to Rho2"
        ))
        
        return links
    
    @staticmethod
    def create_frequency_thickness_links(link_manager: ParameterLinkManager) -> List[ParameterLink]:
        """Create links between frequency and thickness for tuning analysis."""
        links = []
        
        # Tuning thickness is approximately λ/4 = V/(4*f)
        # This is a simplified relationship
        links.append(ParameterLink(
            source_param='wavelet_freq',
            target_param='max_thickness',
            link_type=LinkType.INVERSE,
            factor=2500,  # Assuming V2 = 2500 m/s
            description="Tuning thickness relationship: t_tuning ≈ V/(4*f)"
        ))
        
        return links

# Global link manager instance
_global_link_manager = None

def get_global_link_manager() -> ParameterLinkManager:
    """Get the global parameter link manager instance."""
    global _global_link_manager
    if _global_link_manager is None:
        _global_link_manager = ParameterLinkManager()
    return _global_link_manager

def create_seismic_parameter_links() -> ParameterLinkManager:
    """Create a parameter link manager with common seismic modeling links."""
    link_manager = ParameterLinkManager()
    
    # Add common seismic modeling links
    velocity_links = SeismicLinkPresets.create_velocity_sequence_links(link_manager)
    density_links = SeismicLinkPresets.create_density_sequence_links(link_manager)
    frequency_links = SeismicLinkPresets.create_frequency_thickness_links(link_manager)
    
    for link in velocity_links + density_links + frequency_links:
        link_manager.add_link(link)
    
    return link_manager


