import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import json
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os

class InteractivePlotter:
    """
    Interactive plotting class for real-time visualization with zoom, pan, and export capabilities.
    """
    
    def __init__(self, backend='plotly'):
        """
        Initialize the interactive plotter.
        
        Args:
            backend: 'plotly' for interactive plots, 'matplotlib' for static plots
        """
        self.backend = backend
        self.current_plots = {}  # Store current plot data
        
    def create_ricker_plot(self, 
                          frequency: float,
                          time_length: float = 256.0,
                          dt: float = 0.001,
                          show_spectrum: bool = True,
                          plot_id: str = "ricker_1") -> Dict[str, Any]:
        """
        Create an interactive Ricker wavelet plot.
        
        Args:
            frequency: Dominant frequency in Hz
            time_length: Time length in ms
            dt: Time sampling interval in seconds
            show_spectrum: Whether to show frequency spectrum
            plot_id: Unique identifier for the plot
            
        Returns:
            Dictionary with plot data and metadata
        """
        # Generate wavelet data
        from tools.ricker_tools import create_ricker_wavelet, analyze_wavelet
        
        time_array, wavelet = create_ricker_wavelet(frequency, time_length, dt)
        properties = analyze_wavelet(time_array, wavelet, dt)
        
        if self.backend == 'plotly':
            return self._create_plotly_ricker_plot(
                time_array, wavelet, properties, frequency, show_spectrum, plot_id
            )
        else:
            return self._create_matplotlib_ricker_plot(
                time_array, wavelet, properties, frequency, show_spectrum, plot_id
            )
    
    def _create_plotly_ricker_plot(self, 
                                  time_array: np.ndarray,
                                  wavelet: np.ndarray,
                                  properties: Dict[str, Any],
                                  frequency: float,
                                  show_spectrum: bool,
                                  plot_id: str) -> Dict[str, Any]:
        """Create interactive Plotly plot for Ricker wavelet."""
        
        if show_spectrum:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Time Domain', 'Frequency Domain'),
                vertical_spacing=0.1,
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Time domain plot
            fig.add_trace(
                go.Scatter(
                    x=time_array,
                    y=wavelet,
                    mode='lines',
                    name='Wavelet',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Time:</b> %{x:.2f} ms<br><b>Amplitude:</b> %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Frequency domain plot
            fig.add_trace(
                go.Scatter(
                    x=properties['frequencies'],
                    y=properties['amplitude_spectrum'],
                    mode='lines',
                    name='Amplitude Spectrum',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>Frequency:</b> %{x:.1f} Hz<br><b>Amplitude:</b> %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'Ricker Wavelet (f={frequency} Hz)',
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            
        else:
            # Single time domain plot
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=time_array,
                    y=wavelet,
                    mode='lines',
                    name='Wavelet',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Time:</b> %{x:.2f} ms<br><b>Amplitude:</b> %{y:.4f}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title=f'Ricker Wavelet (f={frequency} Hz)',
                xaxis_title="Time (ms)",
                yaxis_title="Amplitude",
                height=400,
                showlegend=True
            )
        
        # Add zoom and pan controls
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            ),
            dragmode='pan'
        )
        
        # Store plot data
        plot_data = {
            'plot_id': plot_id,
            'type': 'ricker',
            'frequency': frequency,
            'time_length': time_length,
            'dt': dt,
            'time_array': time_array.tolist(),
            'wavelet': wavelet.tolist(),
            'properties': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in properties.items()},
            'figure': fig
        }
        
        self.current_plots[plot_id] = plot_data
        
        return plot_data
    
    def _create_matplotlib_ricker_plot(self, 
                                      time_array: np.ndarray,
                                      wavelet: np.ndarray,
                                      properties: Dict[str, Any],
                                      frequency: float,
                                      show_spectrum: bool,
                                      plot_id: str) -> Dict[str, Any]:
        """Create static matplotlib plot for Ricker wavelet."""
        
        if show_spectrum:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 4))
        
        # Time domain plot
        ax1.plot(time_array, wavelet, 'b-', linewidth=2, label='Wavelet')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Ricker Wavelet (f={frequency} Hz)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        if show_spectrum:
            # Frequency domain plot
            ax2.plot(properties['frequencies'], properties['amplitude_spectrum'], 'r-', linewidth=2)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Amplitude')
            ax2.set_title('Frequency Spectrum')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_file = tempfile.mkstemp(suffix=".png")[1]
        plt.savefig(temp_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store plot data
        plot_data = {
            'plot_id': plot_id,
            'type': 'ricker',
            'frequency': frequency,
            'time_length': time_length,
            'dt': dt,
            'time_array': time_array.tolist(),
            'wavelet': wavelet.tolist(),
            'properties': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in properties.items()},
            'image_path': temp_file
        }
        
        self.current_plots[plot_id] = plot_data
        
        return plot_data
    
    def create_comparison_plot(self, 
                              plot_ids: List[str],
                              comparison_type: str = "overlay") -> Dict[str, Any]:
        """
        Create a comparison plot of multiple wavelets.
        
        Args:
            plot_ids: List of plot IDs to compare
            comparison_type: 'overlay' or 'subplot'
            
        Returns:
            Dictionary with comparison plot data
        """
        if self.backend != 'plotly':
            raise ValueError("Comparison plots require Plotly backend")
        
        # Get plot data
        plots_data = [self.current_plots[pid] for pid in plot_ids if pid in self.current_plots]
        
        if len(plots_data) < 2:
            raise ValueError("Need at least 2 plots for comparison")
        
        if comparison_type == "overlay":
            return self._create_overlay_comparison(plots_data)
        elif comparison_type == "subplot":
            return self._create_subplot_comparison(plots_data)
        else:
            raise ValueError("comparison_type must be 'overlay' or 'subplot'")
    
    def _create_overlay_comparison(self, plots_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create overlay comparison plot."""
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, plot_data in enumerate(plots_data):
            color = colors[i % len(colors)]
            freq = plot_data['frequency']
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data['time_array'],
                    y=plot_data['wavelet'],
                    mode='lines',
                    name=f'f={freq} Hz',
                    line=dict(color=color, width=2),
                    hovertemplate=f'<b>Frequency:</b> {freq} Hz<br><b>Time:</b> %{{x:.2f}} ms<br><b>Amplitude:</b> %{{y:.4f}}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title='Ricker Wavelet Comparison',
            xaxis_title="Time (ms)",
            yaxis_title="Amplitude",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return {
            'type': 'comparison_overlay',
            'figure': fig,
            'plot_ids': [p['plot_id'] for p in plots_data]
        }
    
    def _create_subplot_comparison(self, plots_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create subplot comparison plot."""
        n_plots = len(plots_data)
        cols = min(2, n_plots)
        rows = (n_plots + 1) // 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"f={p['frequency']} Hz" for p in plots_data],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, plot_data in enumerate(plots_data):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data['time_array'],
                    y=plot_data['wavelet'],
                    mode='lines',
                    name=f'f={plot_data["frequency"]} Hz',
                    line=dict(color='blue', width=2),
                    showlegend=False,
                    hovertemplate=f'<b>Frequency:</b> {plot_data["frequency"]} Hz<br><b>Time:</b> %{{x:.2f}} ms<br><b>Amplitude:</b> %{{y:.4f}}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Ricker Wavelet Comparison (Subplots)',
            height=300 * rows,
            showlegend=False
        )
        
        # Update all axes
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(title_text="Time (ms)", row=i, col=j)
                fig.update_yaxes(title_text="Amplitude", row=i, col=j)
        
        return {
            'type': 'comparison_subplot',
            'figure': fig,
            'plot_ids': [p['plot_id'] for p in plots_data]
        }
    
    def export_plot(self, plot_id: str, format: str = "png", width: int = 1200, height: int = 800) -> str:
        """
        Export a plot to a file.
        
        Args:
            plot_id: ID of the plot to export
            format: Export format ('png', 'svg', 'pdf', 'html')
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Path to the exported file
        """
        if plot_id not in self.current_plots:
            raise ValueError(f"Plot {plot_id} not found")
        
        plot_data = self.current_plots[plot_id]
        
        if self.backend == 'plotly' and 'figure' in plot_data:
            fig = plot_data['figure']
            
            if format == "html":
                # Export as interactive HTML
                temp_file = tempfile.mkstemp(suffix=".html")[1]
                fig.write_html(temp_file)
            else:
                # Export as static image
                temp_file = tempfile.mkstemp(suffix=f".{format}")[1]
                fig.write_image(temp_file, width=width, height=height, format=format)
            
            return temp_file
        else:
            # For matplotlib plots, return the existing image path
            return plot_data.get('image_path', '')
    
    def get_plot_data(self, plot_id: str) -> Optional[Dict[str, Any]]:
        """Get plot data by ID."""
        return self.current_plots.get(plot_id)
    
    def list_plots(self) -> List[str]:
        """List all available plot IDs."""
        return list(self.current_plots.keys())
    
    def clear_plots(self):
        """Clear all stored plots."""
        self.current_plots.clear()
    
    def delete_plot(self, plot_id: str):
        """Delete a specific plot."""
        if plot_id in self.current_plots:
            del self.current_plots[plot_id]
