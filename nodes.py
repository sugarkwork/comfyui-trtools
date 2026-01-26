import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict
import folder_paths
import json

# Extra imports for model conversion/download
import requests
import onnx
import csv


# Spandrel for model loading
try:
    from spandrel import ModelLoader, ImageModelDescriptor
    try:
        from spandrel_extra_arches import EXTRA_REGISTRY
        from spandrel import MAIN_REGISTRY
        # Safely add EXTRA_REGISTRY
        try:
            MAIN_REGISTRY.add(*EXTRA_REGISTRY)
        except Exception:
            pass
    except ImportError:
        pass
    HAS_SPANDREL = True
except ImportError:
    HAS_SPANDREL = False

# TensorRT imports
# TensorRT imports
try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
except Exception:
    HAS_TENSORRT = False


# PIL for image processing
from PIL import Image

# Setup logging
logger = logging.getLogger('ComfyUI.TRTUpscaler')
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if HAS_TENSORRT else None

KNOWN_UPSCALERS = {
    "4x_foolhardy_Remacri": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth?download=true",
    "4x-UltraSharp": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth?download=true"
}

KNOWN_TAGGERS = [
    "wd-eva02-large-tagger-v3",
    "wd-vit-large-tagger-v3",
    "wd-v1-4-swinv2-tagger-v2",
    "wd-vit-tagger-v3",
]


KNOWN_MODELS = KNOWN_UPSCALERS # Backwards compatibility for now

# Global cache to store deserialized engines (ICudaEngine)
# Key: engine_path (str), Value: trt.ICudaEngine
GLOBAL_ENGINE_CACHE = {}



class ModelManager:
    """Handles model downloading, ONNX export, and TensorRT engine building."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str, ext: str) -> Path:
        return self.models_dir / f"{model_name}.{ext}"
    
    def download_model(self, model_name: str) -> bool:
        # Check Upscalers
        if model_name in KNOWN_UPSCALERS:
            url = KNOWN_UPSCALERS[model_name]
            dest_path = self.get_model_path(model_name, "pth")
            
            if dest_path.exists():
                return True
                
            logger.info(f"Downloading {model_name} from {url}...")
            return self._download_file(url, dest_path)
            
        # Check Taggers
        if model_name in KNOWN_TAGGERS:
            # Taggers need ONNX and CSV
            base_url = f"https://huggingface.co/SmilingWolf/{model_name}/resolve/main/"
            onnx_url = f"{base_url}model.onnx"
            csv_url = f"{base_url}selected_tags.csv"
            
            onnx_dest = self.get_model_path(model_name, "onnx")
            csv_dest = self.get_model_path(model_name, "csv")
            
            if not onnx_dest.exists():
                logger.info(f"Downloading {model_name} ONNX...")
                if not self._download_file(onnx_url, onnx_dest):
                    return False
            
            if not csv_dest.exists():
                logger.info(f"Downloading {model_name} CSV...")
                if not self._download_file(csv_url, csv_dest):
                    return False
            
            return True

        return False

    def _download_file(self, url: str, dest_path: Path) -> bool:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    def export_to_onnx(self, pth_path: Path, onnx_path: Path):
        """Export PyTorch model to ONNX using Spandrel"""
        if not HAS_SPANDREL:
            raise ImportError("Spandrel is required for model conversion (pip install spandrel)")
            
        logger.info(f"Exporting {pth_path} to ONNX...")
        
        # Load state dict
        device = torch.device("cpu") # Export on CPU to avoid VRAM issues
        sd = torch.load(pth_path, map_location="cpu")
        
        # Fix common state dict issues
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("module."):
                    new_sd[k[7:]] = v
                else:
                    new_sd[k] = v
            sd = new_sd
        
        # Load model using spandrel
        model = ModelLoader().load_from_state_dict(sd)
        model = model.model
        model.eval()
        model.to(device)
        
        # Create dummy input (standard size for export)
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            },
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        )
        logger.info(f"ONNX export completed: {onnx_path}")

    def build_engine(self, onnx_path: Path, engine_path: Path, use_fp16: bool = True):
        """Build TensorRT engine from ONNX file using Python API"""
        if not HAS_TENSORRT:
            raise ImportError("TensorRT is not available")
            
        logger.info(f"Building TensorRT engine for {onnx_path}...")
        
        # Separate logger for building to avoid spam
        BUILD_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(BUILD_LOGGER)
        
        # Create network with explicit batch
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, BUILD_LOGGER)
        config = builder.create_builder_config()
        
        # Parse ONNX
        # Using parse_from_file is better as it handles external data (checking relative paths)
        if not parser.parse_from_file(str(onnx_path)):
            for error in range(parser.num_errors):
                logger.error(f"ONNX Parse Error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX file")

        # Optimization Profile
        input_name = "input"
        # Verify input name matches
        if network.num_inputs > 0:
            input_name = network.get_input(0).name
            
        profile = builder.create_optimization_profile()
        # Shapes for Upscaler (NCHW)
        # Min: Small tile
        # Opt: Standard tile
        # Max: Large tile (up to 512x512 input usually enough for tiled processing)
        profile.set_shape(
            input_name,
            (1, 3, 64, 64),    # min
            (1, 3, 256, 256),  # opt
            (1, 3, 512, 512)   # max
        )
        config.add_optimization_profile(profile)
        
        # Memory pool
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024) # 1GB
        except Exception:
            pass # Older TRT versions might not support this or use different API

        # FP16
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Build engine
        try:
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Engine build returned None")
                
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
                
            logger.info(f"Engine saved to {engine_path}")
            
        except Exception as e:
            logger.error(f"Failed to build engine: {e}")
            raise

    def build_engine_tagger(self, onnx_path: Path, engine_path: Path, use_fp16: bool = True):
        """Build TensorRT engine from ONNX file specifically for Taggers (fixed shape usually 448x448)"""
        if not HAS_TENSORRT:
            raise ImportError("TensorRT is not available")
            
        logger.info(f"Building TensorRT engine (Tagger) for {onnx_path}...")
        
        BUILD_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(BUILD_LOGGER)
        
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, BUILD_LOGGER)
        config = builder.create_builder_config()
        
        if not parser.parse_from_file(str(onnx_path)):
            for error in range(parser.num_errors):
                logger.error(f"ONNX Parse Error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX file")

        input_name = "input"
        if network.num_inputs > 0:
            input_name = network.get_input(0).name
            
        profile = builder.create_optimization_profile()
        # WD 1.4 taggers use 448x448 NCHW (NHWC input is preprocessed to NCHW in the custom logic usually, 
        # but models expect specific layout. Standard ONNX taggers usually Input: [1, 448, 448, 3] or [1, 3, 448, 448]?)
        # trtagger.py sets: (1, 448, 448, 3) 
        # Wait, trtagger.py optimization profile says: (1, 448, 448, 3). 
        # This implies NHWC input for calculation?
        # But convert_model.py for upscaler used NCHW.
        # Let's check trtagger.py preprocess again:
        # img = img[:, :, ::-1] # RGB->BGR
        # The profile in trtagger.py is:
        # profile.set_shape(..., (1, 448, 448, 3), (4, 448, 448, 3), (8, 448, 448, 3))
        # This strongly suggests the ONNX model expects NHWC input.
        
        profile.set_shape(
            input_name,
            (1, 448, 448, 3),    # min
            (1, 448, 448, 3),    # opt
            (4, 448, 448, 3)     # max (support batch up to 4)
        )
        config.add_optimization_profile(profile)
        
        # Memory pool
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024) 
        except Exception:
            pass 

        # FP16
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Build engine
        try:
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Engine build returned None")
                
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
                
            logger.info(f"Engine saved to {engine_path}")
            
        except Exception as e:
            logger.error(f"Failed to build engine: {e}")
            raise

    def ensure_model(self, model_name: str, use_fp16: bool = True) -> Path:
        """Ensure the TensorRT engine exists, creating it if necessary."""
        precision_suffix = "_fp16" if use_fp16 else "_fp32"
        engine_path = self.get_model_path(f"{model_name}{precision_suffix}", "trt")
        
        if engine_path.exists():
            return engine_path
            
        logger.info(f"Model {model_name} (TRT) not found. Checking source...")
        
        # Check if it is a known Tagger
        is_tagger = model_name in KNOWN_TAGGERS
        # Also check if it looks like a tagger from file system? (e.g. CSV presence)
        # For now rely on KNOWN_TAGGERS or if csv exists
        if not is_tagger and (self.get_model_path(model_name, "csv").exists()):
            is_tagger = True

        if is_tagger:
             # Ensure ONNX and CSV
            onnx_path = self.get_model_path(model_name, "onnx")
            csv_path = self.get_model_path(model_name, "csv")
            
            if not onnx_path.exists() or not csv_path.exists():
                self.download_model(model_name)
                
            if not onnx_path.exists():
                 raise FileNotFoundError(f"Tagger ONNX file not found for {model_name}")
            
            self.build_engine_tagger(onnx_path, engine_path, use_fp16)
            
        else:
            # Upscaler path
            # Check for PTH
            pth_path = self.get_model_path(model_name, "pth")
            if not pth_path.exists():
                # Try to download
                if not self.download_model(model_name):
                     raise FileNotFoundError(f"Model file {pth_path} not found and download failed/not available.")
            
            # Check for ONNX or export
            onnx_path = self.get_model_path(model_name, "onnx")
            if not onnx_path.exists():
                self.export_to_onnx(pth_path, onnx_path)
                
            # Build Engine
            self.build_engine(onnx_path, engine_path, use_fp16)
        
        return engine_path



class TRTUpscaler:
    """
    TensorRT-accelerated image upscaler node for ComfyUI.
    Supports batch processing and various upscaling models.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available model files
        models_dir = folder_paths.get_folder_paths("upscale_models")[0]
        
        # Collect all available models (PTH and TRT)
        available_models = set()
        
        # Add built-in known models
        for name in KNOWN_MODELS.keys():
            available_models.add(name)
            
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.trt'):
                    name = f.replace('.trt', '')
                    if name.endswith('_fp16') or name.endswith('_fp32'):
                        name = name[:-5]
                    if name not in KNOWN_TAGGERS: # Separate list for taggers? A bit messy if shared dir
                         available_models.add(name)
                elif f.endswith('.pth'):
                    available_models.add(f[:-4])
        
        model_list = sorted(list(available_models))
        
        if not model_list:
            model_list = ["No models found"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_list,),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Size of tiles for processing large images"
                }),
                "tile_overlap": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap between tiles to reduce seams"
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use FP16 precision (faster but may have slight quality loss)"
                }),
            }
        }

    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    
    def __init__(self):
        # Context is per-execution logic usually, but here we cache one per node instance
        # linked to the shared engine.
        self.context_cache = {}

    
    def upscale(self, image: torch.Tensor, model_name: str, tile_size: int = 512, 
                tile_overlap: int = 32, use_fp16: bool = True) -> Tuple[torch.Tensor]:
        """
        Upscale images using TensorRT acceleration.
        
        Args:
            image: Input images tensor [B, H, W, C]
            model_name: Name of the TRT model to use
            tile_size: Size of tiles for processing
            tile_overlap: Overlap between tiles
            use_fp16: Whether to use FP16 precision
            
        Returns:
            Tuple containing upscaled images tensor
        """
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT is not available. Please install TensorRT.")

            
        if model_name == "No models found":
            raise ValueError("No model selected.")
        
        # Get models directory
        models_dir = folder_paths.get_folder_paths("upscale_models")[0]
        manager = ModelManager(models_dir)
        
        # Ensure engine exists (download/convert if needed)
        try:
            engine_path = manager.ensure_model(model_name, use_fp16)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare model {model_name}: {e}")
        
        engine_path = str(engine_path) # Conversion back to string for consistency
        
        # No manual context management needed when using PyTorch

        

        
        try:
            # Load engine and create context in the active CUDA context
            engine_key = engine_path
            
            # Use global cache for engine
            if engine_key not in GLOBAL_ENGINE_CACHE:
                engine = self._load_engine(engine_path)
                if engine is None:
                    raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
                GLOBAL_ENGINE_CACHE[engine_key] = engine
                
            engine = GLOBAL_ENGINE_CACHE[engine_key]
            
            # Context management: we need a context compatible with this engine.
            # We can cache it per instance.
            if engine_key not in self.context_cache:
                context = engine.create_execution_context()
                if context is None:
                    raise RuntimeError("Failed to create TensorRT execution context. Likely out of GPU memory.")
                self.context_cache[engine_key] = context
            
            context = self.context_cache[engine_key]

            
            # Get scale factor from model name
            scale = self._get_scale_from_model_name(model_name)
            
            # Process batch
            batch_size = image.shape[0]
            results = []
            
            for i in range(batch_size):
                # Extract single image [H, W, C]
                single_image = image[i]
                
                # Convert to numpy and ensure contiguous
                img_np = single_image.cpu().numpy()
                img_np = np.ascontiguousarray(img_np)
                
                # Ensure proper format (0-1 range, RGB)
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                
                # Process with tiling
                upscaled = self._process_image_with_tiles(
                    img_np, engine, context, scale, tile_size, tile_overlap
                )
                
                # Convert back to torch tensor
                upscaled_tensor = torch.from_numpy(upscaled).to(image.device)
                results.append(upscaled_tensor)
            
            # Stack results back into batch
            output = torch.stack(results, dim=0)
            
        except Exception as e:
            raise
        finally:
            # PyTorch manages CUDA context automatically
            pass

        
        return (output,)
    
    def _load_engine(self, engine_path: str):
        """Load TensorRT engine from file"""
        runtime = trt.Runtime(TRT_LOGGER)
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            return runtime.deserialize_cuda_engine(engine_data)
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return None
    
    def _get_scale_from_model_name(self, model_name: str) -> int:
        """Extract scale factor from model name"""
        model_name_lower = model_name.lower()
        if "4x" in model_name_lower:
            return 4
        elif "2x" in model_name_lower:
            return 2
        elif "8x" in model_name_lower:
            return 8
        else:
            return 4
    
    def _process_image_with_tiles(self, img_np: np.ndarray, engine, context, 
                                  scale: int, tile_size: int, overlap: int) -> np.ndarray:
        """Process image using tiled approach for large images"""
        height, width = img_np.shape[:2]
        
        # Calculate output dimensions
        out_height = height * scale
        out_width = width * scale
        
        # Initialize output buffer
        output = np.zeros((out_height, out_width, 3), dtype=np.float32)
        weight_map = np.zeros((out_height, out_width, 3), dtype=np.float32)
        
        # Process tiles
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Extract tile
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                
                tile = img_np[y:y_end, x:x_end]
                tile_h, tile_w = tile.shape[:2]
                
                # Handle small tiles with padding
                min_size = 64
                if tile_h < min_size or tile_w < min_size:
                    pad_h = max(0, min_size - tile_h)
                    pad_w = max(0, min_size - tile_w)
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                
                # Preprocess tile
                tile_input = self._preprocess(tile)
                
                # Run inference
                tile_output = self._infer_tile(tile_input, engine, context, scale)
                
                # Remove padding if applied
                if tile_h < min_size or tile_w < min_size:
                    original_output_h = tile_h * scale
                    original_output_w = tile_w * scale
                    tile_output = tile_output[:original_output_h, :original_output_w]
                
                # Calculate output position
                out_y = y * scale
                out_y_end = y_end * scale
                out_x = x * scale
                out_x_end = x_end * scale
                
                # Create feather mask for blending
                mask = self._create_feather_mask(
                    tile_output.shape[:2],
                    x > 0,
                    x_end < width,
                    y > 0,
                    y_end < height,
                    overlap * scale // 2
                )
                
                # Accumulate output with blending
                output[out_y:out_y_end, out_x:out_x_end] += tile_output * mask[:, :, np.newaxis]
                weight_map[out_y:out_y_end, out_x:out_x_end] += mask[:, :, np.newaxis]
        
        # Normalize by weights
        output = output / np.maximum(weight_map, 1e-8)
        
        # Ensure output is in valid range
        output = np.clip(output, 0, 1)
        
        return output
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for TensorRT (HWC to CHW, add batch dimension)"""
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        # Ensure float32 and contiguous
        image = np.ascontiguousarray(image, dtype=np.float32)
        return image
    
    def _infer_tile(self, tile_input: np.ndarray, engine, context, scale: int) -> np.ndarray:
        """Run inference on a single tile"""
        batch_size, channels, h, w = tile_input.shape
        
        # Get input/output names
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        
        # Set input shape
        context.set_input_shape(input_name, tile_input.shape)
        
        # Calculate output shape
        out_h = h * scale
        out_w = w * scale
        output_shape = (batch_size, channels, out_h, out_w)
        
        # Allocate GPU memory using PyTorch
        try:
            # Create input tensor on device
            d_input = torch.from_numpy(tile_input).contiguous().to(device="cuda", dtype=torch.float32)
            
            # Allocate output tensor on device
            d_output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
            
        except torch.cuda.OutOfMemoryError as e:
             raise RuntimeError(f"GPU memory allocation failed. Try reducing tile size. Error: {e}")
        
        # Create bindings
        # Use data_ptr() to get the GPU address
        bindings = [int(d_input.data_ptr()), int(d_output.data_ptr())]

        # Run inference with newer TensorRT API for better context handling
        try:
            # Set tensor addresses for newer API
            context.set_tensor_address(input_name, int(d_input.data_ptr()))
            context.set_tensor_address(output_name, int(d_output.data_ptr()))
            
            # Execute
            success = context.execute_v2(bindings)
            
        except AttributeError:
             # Fallback
            success = context.execute_v2(bindings)
        except Exception as e:
            success = context.execute_v2(bindings)
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Synchronize to ensure completion
        torch.cuda.synchronize()
        
        # Copy output back to host
        # PyTorch handles transfer
        output = d_output.cpu().numpy()
        
        # Remove batch dimension and convert CHW to HWC
        output = output[0].transpose(1, 2, 0)
        
        # Tensors will be freed automatically by PyTorch when they go out of scope
        return output

    
    def _create_feather_mask(self, shape: tuple, has_left: bool, has_right: bool,
                             has_top: bool, has_bottom: bool, feather_size: int) -> np.ndarray:
        """Create feathering mask for tile blending"""
        h, w = shape
        mask = np.ones((h, w), dtype=np.float32)
        
        if feather_size > 0:
            # Left edge
            if has_left:
                for i in range(min(feather_size, w)):
                    mask[:, i] *= i / feather_size
            
            # Right edge
            if has_right:
                for i in range(min(feather_size, w)):
                    mask[:, -(i+1)] *= i / feather_size
            
            # Top edge
            if has_top:
                for i in range(min(feather_size, h)):
                    mask[i, :] *= i / feather_size
            
            # Bottom edge
            if has_bottom:
                for i in range(min(feather_size, h)):
                    mask[-(i+1), :] *= i / feather_size
        
        return mask
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution if model changes
        return float("nan")


class TRTagger:
    """
    TensorRT-accelerated image tagger node for ComfyUI.
    Uses SmileWolf's WD 1.4 tagger models.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available model files
        models_dir = folder_paths.get_folder_paths("upscale_models")[0]
        
        # Collect available tagger models
        available_models = set()
        
        # Add built-in known taggers
        for name in KNOWN_TAGGERS:
            available_models.add(name)
            
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                # We identify taggers by .csv files generally alongside engines
                if f.endswith('.csv'):
                    name = f[:-4]
                    available_models.add(name)
        
        model_list = sorted(list(available_models))
        
        if not model_list:
            model_list = ["No models found"]
            
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_list,),
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_fp16": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "tag"
    CATEGORY = "image/tagging"
    
    def __init__(self):
        self.context_cache = {}
        self.tags_cache = {}


    def load_tags(self, csv_path):
        tags = []
        general_index = None
        character_index = None
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                tags.append(row[1])
        return tags, general_index, character_index
        
    def _preprocess(self, img: Image.Image, h=448):
        ratio = float(h) / max(img.size)
        new_size = tuple([int(x * ratio) for x in img.size])
        img = img.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (h, h), (255, 255, 255))
        square.paste(img, ((h - new_size[0]) // 2, (h - new_size[1]) // 2))
        img = np.array(square).astype(np.float32)
        # RGB -> BGR is done in trtagger.py. ComfyUI gives RGB. 
        img = img[:, :, ::-1] 
        return img

    def tag(self, image: torch.Tensor, model_name: str, threshold: float = 0.35, 
            character_threshold: float = 0.85, use_fp16: bool = True) -> Tuple[str]:
            
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT is not available. Please install TensorRT.")

            
        if model_name == "No models found":
            raise ValueError("No model selected.")

        # Get models directory
        models_dir = folder_paths.get_folder_paths("upscale_models")[0]
        manager = ModelManager(models_dir)
        
        # Ensure engine exists
        try:
            engine_path = manager.ensure_model(model_name, use_fp16)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare model {model_name}: {e}")
            
        engine_path = str(engine_path)
        csv_path = str(manager.get_model_path(model_name, "csv"))
        
        # Load tags if not cached
        if model_name not in self.tags_cache:
            if not os.path.exists(csv_path):
                 raise FileNotFoundError(f"Tag CSV not found: {csv_path}")
            self.tags_cache[model_name] = self.load_tags(csv_path)
            
        tags, general_index, character_index = self.tags_cache[model_name]
        

        try:
            # Load Engine using Global Cache
            if engine_path not in GLOBAL_ENGINE_CACHE:
                runtime = trt.Runtime(TRT_LOGGER)
                with open(engine_path, 'rb') as f:
                    engine = runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
                GLOBAL_ENGINE_CACHE[engine_path] = engine
            
            engine = GLOBAL_ENGINE_CACHE[engine_path]
            
            # Context per instance
            if engine_path not in self.context_cache:
                 self.context_cache[engine_path] = engine.create_execution_context()
                 
            context = self.context_cache[engine_path]

            
            input_name = engine.get_tensor_name(0)
            output_name = engine.get_tensor_name(1)

            final_tags_list = []
            
            batch_size = image.shape[0]
            for i in range(batch_size):
                img_tensor = image[i] # H, W, C
                img_np_01 = img_tensor.cpu().numpy()
                img_pil = Image.fromarray((img_np_01 * 255).astype(np.uint8))
                
                # Preprocess
                processed_img = self._preprocess(img_pil, h=448) # Returns H, W, C (BGR), float32
                
                # Expand to batch
                batch_input = np.expand_dims(processed_img, axis=0) # 1, 448, 448, 3
                batch_input = np.ascontiguousarray(batch_input)
                
                # Inference with PyTorch management
                try:
                    # Input to GPU
                    d_input = torch.from_numpy(batch_input).contiguous().to(device="cuda", dtype=torch.float32)
                    
                    # Output buffer on GPU
                    output_shape = engine.get_tensor_shape(output_name) # (Batch, Tags)
                    num_tags = output_shape[1]
                    d_output = torch.empty((1, num_tags), device="cuda", dtype=torch.float32)
                    
                    bindings = [int(d_input.data_ptr()), int(d_output.data_ptr())]
                    
                    context.set_input_shape(input_name, batch_input.shape)
                    
                    # Execute
                    context.execute_v2(bindings)
                    
                    # Sync
                    torch.cuda.synchronize()
                    
                    # Copy back
                    output = d_output.cpu().numpy()
                    
                except Exception as e:
                     raise RuntimeError(f"Inference failed: {e}")

                probs = output[0]
                
                # Postprocess
                result = list(zip(tags, probs))
                general = [item for item in result[general_index:character_index] if item[1] > threshold]
                character = [item for item in result[character_index:] if item[1] > character_threshold]
                all_tags = character + general
                res = ", ".join(item[0] for item in all_tags)
                final_tags_list.append(res)
                
            return (", ".join(final_tags_list),) 
            
        finally:
            pass


NODE_CLASS_MAPPINGS = {
    "TRTUpscaler": TRTUpscaler,
    "TRTagger": TRTagger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TRTUpscaler": "TRT Upscaler",
    "TRTagger": "TRT Tagger"
}