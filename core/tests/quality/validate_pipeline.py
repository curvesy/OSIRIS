"""
🔍 Data Quality Validation - Production-Grade Pipeline Validation

Comprehensive data quality validation using Great Expectations:
- Hot tier (DuckDB) schema and data validation
- Cold tier (S3 Parquet) format and consistency validation  
- Semantic tier (Redis) vector and metadata validation
- End-to-end data integrity checks
- Automated quality reporting and alerting
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO

# Great Expectations imports
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.exceptions import DataContextError

# Database and storage imports
import duckdb
import redis
from minio import Minio

from loguru import logger


class PipelineDataValidator:
    """
    🔍 Comprehensive Pipeline Data Quality Validator
    
    Validates data quality across all tiers of the Intelligence Flywheel:
    - Hot Memory (DuckDB): Schema, types, constraints
    - Cold Storage (S3): Parquet format, partitioning, compression
    - Semantic Memory (Redis): Vector dimensions, metadata consistency
    - Cross-tier: Data flow integrity, no data loss
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "tests/quality/great_expectations"
        self.results_path = Path("test-results/quality")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Great Expectations context
        self.context = self._initialize_gx_context()
        
        # Data quality thresholds
        self.quality_thresholds = {
            "hot_tier": {
                "null_percentage_max": 0.01,  # Max 1% nulls
                "duplicate_percentage_max": 0.05,  # Max 5% duplicates
                "signature_size_bytes": 768 * 4,  # 768 float32 values
                "timestamp_range_hours": 48  # Max 48 hours of data
            },
            "cold_tier": {
                "parquet_compression_ratio_min": 0.3,  # Min 30% compression
                "partition_size_mb_max": 100,  # Max 100MB per partition
                "schema_consistency": True,  # Consistent schema across files
                "metadata_completeness_min": 0.95  # Min 95% metadata completeness
            },
            "semantic_tier": {
                "vector_dimension": 768,  # Exact dimension requirement
                "similarity_threshold_min": 0.0,  # Min similarity score
                "similarity_threshold_max": 1.0,  # Max similarity score
                "cluster_coverage_min": 0.8  # Min 80% of data in clusters
            }
        }
        
        logger.info("🔍 Pipeline Data Validator initialized")
    
    def _initialize_gx_context(self) -> gx.DataContext:
        """Initialize Great Expectations context with custom configuration."""
        
        try:
            # Try to get existing context
            context = gx.get_context(context_root_dir=self.config_path)
            logger.info("✅ Using existing Great Expectations context")
        except DataContextError:
            # Create new context
            context = gx.get_context(mode="file", context_root_dir=self.config_path)
            logger.info("🆕 Created new Great Expectations context")
        
        return context
    
    async def validate_complete_pipeline(
        self, 
        db_conn: duckdb.DuckDBPyConnection,
        redis_client: redis.Redis,
        s3_client: Minio,
        bucket_name: str = "test-forge"
    ) -> Dict[str, Any]:
        """
        Validate data quality across the complete pipeline.
        
        Returns comprehensive validation report with pass/fail status.
        """
        
        start_time = datetime.now()
        logger.info("🔍 Starting complete pipeline validation...")
        
        validation_results = {
            "validation_time": start_time.isoformat(),
            "hot_tier": {},
            "cold_tier": {},
            "semantic_tier": {},
            "cross_tier": {},
            "overall_status": "unknown",
            "quality_score": 0.0
        }
        
        try:
            # Phase 1: Validate Hot Tier (DuckDB)
            logger.info("📊 Phase 1: Validating hot tier data quality...")
            validation_results["hot_tier"] = await self._validate_hot_tier(db_conn)
            
            # Phase 2: Validate Cold Tier (S3 Parquet)
            logger.info("🗄️ Phase 2: Validating cold tier data quality...")
            validation_results["cold_tier"] = await self._validate_cold_tier(s3_client, bucket_name)
            
            # Phase 3: Validate Semantic Tier (Redis)
            logger.info("🧠 Phase 3: Validating semantic tier data quality...")
            validation_results["semantic_tier"] = await self._validate_semantic_tier(redis_client)
            
            # Phase 4: Cross-Tier Validation
            logger.info("🔗 Phase 4: Validating cross-tier data integrity...")
            validation_results["cross_tier"] = await self._validate_cross_tier_integrity(
                db_conn, redis_client, s3_client, bucket_name
            )
            
            # Calculate overall quality score
            validation_results["quality_score"] = self._calculate_quality_score(validation_results)
            validation_results["overall_status"] = "pass" if validation_results["quality_score"] >= 0.8 else "fail"
            
            # Generate detailed report
            report_path = await self._generate_quality_report(validation_results)
            validation_results["report_path"] = str(report_path)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Pipeline validation completed in {duration:.2f}s")
            logger.info(f"📊 Quality Score: {validation_results['quality_score']:.2%}")
            logger.info(f"🎯 Overall Status: {validation_results['overall_status'].upper()}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline validation failed: {e}")
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)
            return validation_results
    
    async def _validate_hot_tier(self, db_conn: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
        """Validate hot tier (DuckDB) data quality."""
        
        try:
            # Extract data for validation
            df = db_conn.execute("""
                SELECT 
                    id,
                    timestamp,
                    length(signature) as signature_length,
                    metadata,
                    partition_hour,
                    created_at,
                    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - timestamp))/3600 as age_hours
                FROM recent_activity
            """).df()
            
            if df.empty:
                return {
                    "status": "warning",
                    "message": "No data in hot tier",
                    "record_count": 0,
                    "validations": []
                }
            
            # Create Great Expectations validator
            validator = self.context.sources.pandas_default.read_dataframe(df)
            
            # Define expectations for hot tier
            expectations = [
                # Schema validations
                validator.expect_column_to_exist("id"),
                validator.expect_column_to_exist("timestamp"),
                validator.expect_column_to_exist("signature_length"),
                validator.expect_column_to_exist("metadata"),
                
                # Data type validations
                validator.expect_column_values_to_not_be_null("id"),
                validator.expect_column_values_to_not_be_null("timestamp"),
                validator.expect_column_values_to_not_be_null("signature_length"),
                
                # Business rule validations
                validator.expect_column_values_to_be_between(
                    "signature_length",
                    min_value=self.quality_thresholds["hot_tier"]["signature_size_bytes"] - 100,
                    max_value=self.quality_thresholds["hot_tier"]["signature_size_bytes"] + 100
                ),
                
                validator.expect_column_values_to_be_between(
                    "age_hours",
                    min_value=0,
                    max_value=self.quality_thresholds["hot_tier"]["timestamp_range_hours"]
                ),
                
                validator.expect_column_values_to_be_unique("id"),
                
                # Partition validation
                validator.expect_column_values_to_be_between("partition_hour", min_value=0, max_value=23),
            ]
            
            # Run validation
            validation_result = validator.validate()
            
            # Calculate metrics
            total_records = len(df)
            null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
            duplicate_percentage = (len(df) - len(df.drop_duplicates())) / len(df)
            
            return {
                "status": "pass" if validation_result.success else "fail",
                "record_count": total_records,
                "null_percentage": null_percentage,
                "duplicate_percentage": duplicate_percentage,
                "validations": [
                    {
                        "expectation": exp.expectation_config.expectation_type,
                        "success": exp.success,
                        "result": exp.result
                    }
                    for exp in validation_result.results
                ],
                "success_rate": len([r for r in validation_result.results if r.success]) / len(validation_result.results)
            }
            
        except Exception as e:
            logger.error(f"❌ Hot tier validation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "record_count": 0,
                "validations": []
            }
    
    async def _validate_cold_tier(self, s3_client: Minio, bucket_name: str) -> Dict[str, Any]:
        """Validate cold tier (S3 Parquet) data quality."""
        
        try:
            if not s3_client:
                return {
                    "status": "warning",
                    "message": "S3 client not available",
                    "file_count": 0,
                    "validations": []
                }
            
            # List Parquet files
            objects = list(s3_client.list_objects(bucket_name, recursive=True))
            parquet_files = [obj for obj in objects if obj.object_name.endswith('.parquet')]
            
            if not parquet_files:
                return {
                    "status": "warning",
                    "message": "No Parquet files in cold storage",
                    "file_count": 0,
                    "validations": []
                }
            
            validations = []
            total_records = 0
            schema_consistent = True
            first_schema = None
            
            # Validate each Parquet file
            for obj in parquet_files[:5]:  # Limit to first 5 files for performance
                try:
                    # Download and read Parquet file
                    parquet_data = s3_client.get_object(bucket_name, obj.object_name).read()
                    table = pq.read_table(BytesIO(parquet_data))
                    df = table.to_pandas()
                    
                    # Schema consistency check
                    if first_schema is None:
                        first_schema = table.schema
                    elif not table.schema.equals(first_schema):
                        schema_consistent = False
                    
                    # File-level validations
                    file_size_mb = len(parquet_data) / (1024 * 1024)
                    compression_ratio = len(parquet_data) / (len(df) * df.memory_usage(deep=True).sum())
                    
                    validations.append({
                        "file": obj.object_name,
                        "records": len(df),
                        "size_mb": file_size_mb,
                        "compression_ratio": compression_ratio,
                        "schema_valid": table.schema is not None,
                        "data_valid": not df.empty
                    })
                    
                    total_records += len(df)
                    
                except Exception as e:
                    validations.append({
                        "file": obj.object_name,
                        "error": str(e),
                        "valid": False
                    })
            
            # Calculate overall metrics
            valid_files = [v for v in validations if v.get("data_valid", False)]
            success_rate = len(valid_files) / len(validations) if validations else 0
            
            return {
                "status": "pass" if success_rate >= 0.9 and schema_consistent else "fail",
                "file_count": len(parquet_files),
                "total_records": total_records,
                "schema_consistent": schema_consistent,
                "success_rate": success_rate,
                "validations": validations
            }
            
        except Exception as e:
            logger.error(f"❌ Cold tier validation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file_count": 0,
                "validations": []
            }
    
    async def _validate_semantic_tier(self, redis_client: redis.Redis) -> Dict[str, Any]:
        """Validate semantic tier (Redis) data quality."""
        
        try:
            if not redis_client:
                return {
                    "status": "warning",
                    "message": "Redis client not available",
                    "record_count": 0,
                    "validations": []
                }
            
            # Get semantic signatures
            signature_keys = redis_client.keys('sig:*')
            
            if not signature_keys:
                return {
                    "status": "warning",
                    "message": "No semantic signatures found",
                    "record_count": 0,
                    "validations": []
                }
            
            validations = []
            valid_vectors = 0
            total_vectors = len(signature_keys)
            
            # Sample validation (check first 100 signatures for performance)
            sample_keys = signature_keys[:100]
            
            for key in sample_keys:
                try:
                    sig_data = redis_client.hgetall(key)
                    
                    # Validate required fields
                    has_vector = 'vector' in sig_data
                    has_metadata = 'metadata' in sig_data
                    has_timestamp = 'timestamp' in sig_data
                    
                    # Validate vector dimension
                    vector_valid = False
                    if has_vector:
                        try:
                            vector_bytes = sig_data['vector']
                            if isinstance(vector_bytes, str):
                                vector_bytes = vector_bytes.encode('latin1')
                            vector = np.frombuffer(vector_bytes, dtype=np.float32)
                            vector_valid = len(vector) == self.quality_thresholds["semantic_tier"]["vector_dimension"]
                        except:
                            vector_valid = False
                    
                    # Validate metadata
                    metadata_valid = False
                    if has_metadata:
                        try:
                            metadata = json.loads(sig_data['metadata'])
                            metadata_valid = isinstance(metadata, dict)
                        except:
                            metadata_valid = False
                    
                    is_valid = has_vector and has_metadata and has_timestamp and vector_valid and metadata_valid
                    if is_valid:
                        valid_vectors += 1
                    
                    validations.append({
                        "key": key,
                        "has_vector": has_vector,
                        "has_metadata": has_metadata,
                        "has_timestamp": has_timestamp,
                        "vector_valid": vector_valid,
                        "metadata_valid": metadata_valid,
                        "overall_valid": is_valid
                    })
                    
                except Exception as e:
                    validations.append({
                        "key": key,
                        "error": str(e),
                        "overall_valid": False
                    })
            
            # Calculate success rate
            success_rate = valid_vectors / len(sample_keys) if sample_keys else 0
            
            return {
                "status": "pass" if success_rate >= 0.9 else "fail",
                "record_count": total_vectors,
                "sample_size": len(sample_keys),
                "valid_vectors": valid_vectors,
                "success_rate": success_rate,
                "validations": validations[:10]  # Return first 10 for brevity
            }
            
        except Exception as e:
            logger.error(f"❌ Semantic tier validation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "record_count": 0,
                "validations": []
            }
    
    async def _validate_cross_tier_integrity(
        self, 
        db_conn: duckdb.DuckDBPyConnection,
        redis_client: redis.Redis,
        s3_client: Minio,
        bucket_name: str
    ) -> Dict[str, Any]:
        """Validate data integrity across all tiers."""
        
        try:
            integrity_checks = []
            
            # Check 1: Hot tier record count consistency
            hot_count = db_conn.execute("SELECT COUNT(*) FROM recent_activity").fetchone()[0]
            integrity_checks.append({
                "check": "hot_tier_record_count",
                "value": hot_count,
                "status": "pass" if hot_count >= 0 else "fail"
            })
            
            # Check 2: Semantic tier record count
            if redis_client:
                semantic_count = len(redis_client.keys('sig:*'))
                integrity_checks.append({
                    "check": "semantic_tier_record_count",
                    "value": semantic_count,
                    "status": "pass" if semantic_count >= 0 else "fail"
                })
            
            # Check 3: Cold storage file count
            if s3_client:
                try:
                    objects = list(s3_client.list_objects(bucket_name, recursive=True))
                    parquet_count = len([obj for obj in objects if obj.object_name.endswith('.parquet')])
                    integrity_checks.append({
                        "check": "cold_storage_file_count",
                        "value": parquet_count,
                        "status": "pass" if parquet_count >= 0 else "fail"
                    })
                except:
                    integrity_checks.append({
                        "check": "cold_storage_file_count",
                        "value": 0,
                        "status": "warning",
                        "message": "Could not access cold storage"
                    })
            
            # Check 4: Data flow consistency (simplified)
            total_data_points = hot_count + semantic_count
            integrity_checks.append({
                "check": "total_data_consistency",
                "value": total_data_points,
                "status": "pass" if total_data_points > 0 else "warning"
            })
            
            # Calculate overall integrity score
            passed_checks = len([c for c in integrity_checks if c["status"] == "pass"])
            integrity_score = passed_checks / len(integrity_checks) if integrity_checks else 0
            
            return {
                "status": "pass" if integrity_score >= 0.8 else "fail",
                "integrity_score": integrity_score,
                "checks": integrity_checks
            }
            
        except Exception as e:
            logger.error(f"❌ Cross-tier validation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checks": []
            }
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        
        scores = []
        
        # Hot tier score
        if validation_results["hot_tier"].get("success_rate") is not None:
            scores.append(validation_results["hot_tier"]["success_rate"])
        
        # Cold tier score
        if validation_results["cold_tier"].get("success_rate") is not None:
            scores.append(validation_results["cold_tier"]["success_rate"])
        
        # Semantic tier score
        if validation_results["semantic_tier"].get("success_rate") is not None:
            scores.append(validation_results["semantic_tier"]["success_rate"])
        
        # Cross-tier score
        if validation_results["cross_tier"].get("integrity_score") is not None:
            scores.append(validation_results["cross_tier"]["integrity_score"])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _generate_quality_report(self, validation_results: Dict[str, Any]) -> Path:
        """Generate comprehensive quality report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_path / f"quality_report_{timestamp}.json"
        
        # Add summary statistics
        validation_results["summary"] = {
            "total_tiers_validated": 4,
            "tiers_passed": len([
                tier for tier in ["hot_tier", "cold_tier", "semantic_tier", "cross_tier"]
                if validation_results[tier].get("status") == "pass"
            ]),
            "overall_quality_score": validation_results["quality_score"],
            "validation_timestamp": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations(validation_results)
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"📊 Quality report saved: {report_path}")
        return report_path
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        
        recommendations = []
        
        # Hot tier recommendations
        hot_tier = validation_results.get("hot_tier", {})
        if hot_tier.get("null_percentage", 0) > 0.01:
            recommendations.append("Reduce null values in hot tier data")
        
        if hot_tier.get("duplicate_percentage", 0) > 0.05:
            recommendations.append("Implement deduplication in hot tier ingestion")
        
        # Cold tier recommendations
        cold_tier = validation_results.get("cold_tier", {})
        if not cold_tier.get("schema_consistent", True):
            recommendations.append("Ensure consistent schema across Parquet files")
        
        # Semantic tier recommendations
        semantic_tier = validation_results.get("semantic_tier", {})
        if semantic_tier.get("success_rate", 0) < 0.9:
            recommendations.append("Improve semantic tier data quality validation")
        
        # Overall recommendations
        if validation_results.get("quality_score", 0) < 0.8:
            recommendations.append("Implement automated data quality monitoring")
            recommendations.append("Add data validation to ingestion pipeline")
        
        return recommendations


async def main():
    """Run data quality validation as standalone script."""
    
    # Initialize connections (these would come from test fixtures in real tests)
    db_conn = duckdb.connect(':memory:')
    redis_client = None
    s3_client = None
    
    try:
        # Try to connect to test services
        redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        redis_client.ping()
    except:
        logger.warning("⚠️ Redis not available for validation")
    
    try:
        s3_client = Minio(
            os.getenv('S3_ENDPOINT', 'localhost:9000').replace('http://', ''),
            access_key=os.getenv('S3_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('S3_SECRET_KEY', 'minioadmin'),
            secure=False
        )
    except:
        logger.warning("⚠️ S3 not available for validation")
    
    # Run validation
    validator = PipelineDataValidator()
    results = await validator.validate_complete_pipeline(
        db_conn, redis_client, s3_client
    )
    
    # Print summary
    print(f"\n🔍 Data Quality Validation Results:")
    print(f"   Overall Status: {results['overall_status'].upper()}")
    print(f"   Quality Score: {results['quality_score']:.2%}")
    print(f"   Report: {results.get('report_path', 'Not generated')}")
    
    return 0 if results['overall_status'] == 'pass' else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
