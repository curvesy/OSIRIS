"""
Unit tests for k-NN Index
========================

Tests the core functionality of our vector search index.
"""

import unittest
import numpy as np

from .knn_index import KNNIndex, KNNConfig


class TestKNNIndex(unittest.TestCase):
    """Test suite for KNNIndex."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 4
        self.index = KNNIndex(embedding_dim=self.embedding_dim)
        
        # Create test vectors with known relationships
        self.test_vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],  # id_a: unit vector in x
            [0.0, 1.0, 0.0, 0.0],  # id_b: unit vector in y
            [0.9, 0.1, 0.0, 0.0],  # id_c: close to id_a
            [-1.0, 0.0, 0.0, 0.0]  # id_d: opposite to id_a
        ], dtype=np.float32)
        
        self.test_ids = ["id_a", "id_b", "id_c", "id_d"]
        self.index.add(self.test_vectors, self.test_ids)

    def test_add_and_len(self):
        """Test adding vectors and checking index size."""
        self.assertEqual(len(self.index), 4)
        
        # Add more vectors
        new_vectors = np.random.rand(2, self.embedding_dim).astype(np.float32)
        new_ids = ["id_e", "id_f"]
        self.index.add(new_vectors, new_ids)
        
        self.assertEqual(len(self.index), 6)

    def test_search_exact_match(self):
        """Test searching for an exact match."""
        # Query with vector identical to id_a
        query = np.array([1.0, 0.0, 0.0, 0.0])
        results = self.index.search(query, k=2)
        
        self.assertEqual(len(results), 2)
        
        # First result should be exact match
        self.assertEqual(results[0][0], "id_a")
        self.assertAlmostEqual(results[0][1], 0.0, places=5)
        
        # Second should be id_c (closest to id_a)
        self.assertEqual(results[1][0], "id_c")

    def test_search_k_limit(self):
        """Test that k is properly limited."""
        query = np.random.rand(self.embedding_dim)
        
        # Request more neighbors than exist
        results = self.index.search(query, k=10)
        self.assertEqual(len(results), 4)  # Only 4 vectors in index

    def test_search_empty_index(self):
        """Test searching an empty index."""
        empty_index = KNNIndex(embedding_dim=self.embedding_dim)
        query = np.random.rand(self.embedding_dim)
        
        results = empty_index.search(query, k=5)
        self.assertEqual(len(results), 0)

    def test_dimension_validation(self):
        """Test dimension mismatch handling."""
        # Wrong dimension vector
        wrong_dim = np.array([[1.0, 0.0]])  # 2D instead of 4D
        
        with self.assertRaises(ValueError) as ctx:
            self.index.add(wrong_dim, ["id_x"])
        
        self.assertIn("Wrong dimension", str(ctx.exception))

    def test_id_count_validation(self):
        """Test ID count mismatch handling."""
        vectors = np.random.rand(2, self.embedding_dim)
        
        with self.assertRaises(ValueError) as ctx:
            self.index.add(vectors, ["only_one_id"])
        
        self.assertIn("Mismatch", str(ctx.exception))

    def test_duplicate_id_validation(self):
        """Test duplicate ID detection."""
        vectors = np.random.rand(1, self.embedding_dim)
        
        with self.assertRaises(ValueError) as ctx:
            self.index.add(vectors, ["id_a"])  # id_a already exists
        
        self.assertIn("already exists", str(ctx.exception))

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid metric
        with self.assertRaises(ValueError):
            KNNConfig(metric='invalid')
        
        # Invalid backend
        with self.assertRaises(ValueError):
            KNNConfig(backend='invalid')

    def test_cosine_similarity(self):
        """Test cosine similarity metric."""
        config = KNNConfig(metric='cosine')
        index = KNNIndex(self.embedding_dim, config)
        
        # Add normalized vectors
        vectors = self.test_vectors / np.linalg.norm(self.test_vectors, axis=1, keepdims=True)
        index.add(vectors, self.test_ids)
        
        # Search with normalized query
        query = np.array([1.0, 0.0, 0.0, 0.0])
        results = index.search(query, k=2)
        
        self.assertEqual(results[0][0], "id_a")


if __name__ == '__main__':
    unittest.main()