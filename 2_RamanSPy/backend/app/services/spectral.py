import ramanspy as rp
import numpy as np
import io
import logging

logger = logging.getLogger(__name__)

class SpectralService:
    @staticmethod
    def load_witec(file_path: str):
        try:
            return rp.load.witec(file_path)
        except Exception as e:
            logger.error(f"Failed to load WITec file: {e}")
            raise e

    @staticmethod
    def load_renishaw(file_path: str):
        try:
            return rp.load.renishaw(file_path)
        except Exception as e:
            logger.error(f"Failed to load Renishaw file: {e}")
            raise e
            
    @staticmethod
    def load_numpy(file_path: str):
        try:
            arr = np.load(file_path)
            return rp.Spectrum(arr)
        except Exception as e:
             logger.error(f"Failed to load Numpy file: {e}")
             raise e

    @staticmethod
    def get_preview_data(data, max_points=2000):
        # Downsample or limit points for preview
        if isinstance(data, rp.Spectrum):
             y = data.spectral_data
             x = data.spectral_axis
             if len(x) > max_points:
                 indices = np.linspace(0, len(x)-1, max_points, dtype=int)
                 x = x[indices]
                 y = y[indices]
             return {"x": x.tolist(), "y": y.tolist(), "type": "spectrum"}
             
        elif isinstance(data, rp.SpectralContainer):
            # Return first few spectra
             y = data[:5].spectral_data
             x = data.spectral_axis
             if len(x) > max_points:
                 indices = np.linspace(0, len(x)-1, max_points, dtype=int)
                 x = x[indices]
                 y = y[:, indices]
             return {"x": x.tolist(), "y": y.tolist(), "type": "container", "count": len(data)}
             
        return None
