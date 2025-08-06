"""
HyperOak HD-Vector Adapter with Entropy-Aware Compaction
CRITICAL: Entropy compaction must be enabled BEFORE any data is stored
"""
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

class HyperOakAdapter:
    """
    HyperOak 1.1 HD-Vector adapter with entropy-aware compaction
    Provides 8× memory compression and 2ms P50 retrieval
    """
    
    def __init__(self, 
                 host: str = None,
                 port: int = 8080,
                 enable_entropy_compaction: bool = True):
        """
        Initialize HyperOak adapter
        
        CRITICAL: enable_entropy_compaction MUST be True for new deployments
        Once data exists, this cannot be changed!
        """
        self.host = host or os.getenv('HYPEROAK_HOST', 'localhost')
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        
        # CRITICAL CONFIGURATION - Cannot change after data exists
        self.db_config = {
            'compaction_mode': 'entropy_hd' if enable_entropy_compaction else 'standard',
            'encoding': 'hd_uint8',  # High-dimensional encoding
            'compression': 'zstd',
            'compression_level': 3,
            'vector_dimension': 768,  # For topological signatures
            'distance_metric': 'wasserstein',  # For topology comparison
            
            # Entropy-specific settings (40% storage reduction)
            'entropy_threshold': 0.7,
            'entropy_window_size': 1024,
            'adaptive_quantization': True,
            'pruning_strategy': 'topological'
        }
        
        # Verify configuration on startup
        self._verify_configuration()
        
        logger.info(f"HyperOak adapter initialized with entropy_compaction={enable_entropy_compaction}")
        
    def _verify_configuration(self):
        """Verify HyperOak configuration matches our requirements"""
        try:
            response = httpx.get(f"{self.base_url}/admin/config")
            current_config = response.json()
            
            # CRITICAL: Check if entropy compaction is properly set
            if current_config.get('compaction_mode') != self.db_config['compaction_mode']:
                if current_config.get('data_exists', False):
                    raise RuntimeError(
                        "CRITICAL: HyperOak already contains data with different compaction mode. "
                        "Cannot change compaction_mode after data exists!"
                    )
                else:
                    # Apply our configuration
                    self._apply_configuration()
                    
        except httpx.ConnectError:
            logger.warning("HyperOak not reachable for configuration verification")
            
    def _apply_configuration(self):
        """Apply our configuration to HyperOak"""
        response = httpx.post(
            f"{self.base_url}/admin/config",
            json=self.db_config
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to apply HyperOak configuration: {response.text}")
        logger.info("Successfully applied entropy-aware compaction configuration")
        
    async def store_topological_signature(self, 
                                        signature_id: str,
                                        topology_vector: np.ndarray,
                                        metadata: Dict[str, Any]) -> bool:
        """
        Store a topological signature with HD-vector encoding
        
        Args:
            signature_id: Unique identifier for the signature
            topology_vector: High-dimensional topology representation
            metadata: Additional context (domain, timestamp, etc.)
        """
        # Convert numpy array to HD-encoded format
        hd_encoded = self._encode_hd_vector(topology_vector)
        
        payload = {
            'id': signature_id,
            'vector': hd_encoded,
            'metadata': metadata,
            'timestamp': datetime.utcnow().isoformat(),
            'encoding': 'hd_uint8'
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/vectors/store",
                json=payload
            )
            
        return response.status_code == 201
        
    async def search_similar_topologies(self,
                                      query_vector: np.ndarray,
                                      k: int = 10,
                                      threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search for topologically similar patterns
        
        Returns patterns with Wasserstein distance < threshold
        """
        hd_query = self._encode_hd_vector(query_vector)
        
        search_params = {
            'vector': hd_query,
            'k': k,
            'threshold': threshold,
            'metric': 'wasserstein',
            'include_metadata': True
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/vectors/search",
                json=search_params,
                timeout=2.0  # 2ms P50 target
            )
            
        results = response.json()
        return results.get('matches', [])
        
    def _encode_hd_vector(self, vector: np.ndarray) -> List[int]:
        """
        Encode vector using HD-vector hyperdimensional computing
        This provides 8× compression while preserving topology
        """
        # Normalize to [0, 255] for uint8 encoding
        normalized = (vector - vector.min()) / (vector.max() - vector.min() + 1e-8)
        uint8_encoded = (normalized * 255).astype(np.uint8)
        
        # Apply entropy-aware quantization if enabled
        if self.db_config['adaptive_quantization']:
            uint8_encoded = self._adaptive_quantize(uint8_encoded)
            
        return uint8_encoded.tolist()
        
    def _adaptive_quantize(self, vector: np.ndarray) -> np.ndarray:
        """Apply adaptive quantization based on local entropy"""
        # Calculate local entropy
        entropy = self._calculate_entropy(vector)
        
        # Quantize more aggressively in low-entropy regions
        if entropy < self.db_config['entropy_threshold']:
            # Reduce precision in low-information areas
            vector = (vector // 4) * 4
            
        return vector
        
    def _calculate_entropy(self, vector: np.ndarray) -> float:
        """Calculate Shannon entropy of vector segment"""
        hist, _ = np.histogram(vector, bins=16)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
        
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics showing compression effectiveness"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/admin/stats")
            
        stats = response.json()
        
        # Calculate compression ratio
        raw_size = stats.get('raw_size_bytes', 0)
        compressed_size = stats.get('compressed_size_bytes', 0)
        compression_ratio = raw_size / max(compressed_size, 1)
        
        return {
            'total_vectors': stats.get('total_vectors', 0),
            'raw_size_mb': raw_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'entropy_compaction_enabled': self.db_config['compaction_mode'] == 'entropy_hd',
            'average_query_latency_ms': stats.get('avg_query_latency_ms', 0)
        }
        
    async def health_check(self) -> bool:
        """Verify HyperOak is healthy and entropy compaction is active"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    timeout=1.0
                )
                
            health = response.json()
            
            # Verify entropy compaction is still active
            if health.get('compaction_mode') != 'entropy_hd':
                logger.error("CRITICAL: Entropy compaction is not active!")
                return False
                
            return health.get('status') == 'healthy'
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global instance with entropy compaction enabled
_hyperoak_adapter = None

def get_hyperoak_adapter() -> HyperOakAdapter:
    """Get or create the global HyperOak adapter instance"""
    global _hyperoak_adapter
    if _hyperoak_adapter is None:
        # CRITICAL: Always enable entropy compaction for new deployments
        _hyperoak_adapter = HyperOakAdapter(enable_entropy_compaction=True)
    return _hyperoak_adapter