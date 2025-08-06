"""
Data processing workflows for AURA Intelligence.

Implements durable data processing pipelines using Temporal
and atomic components for validation, transformation, and enrichment.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import timedelta, timezone

from aura_common.atomic.processors import (
    TextPreprocessor,
    PreprocessorConfig,
    DataValidator,
    ValidationConfig,
    DataTransformer,
    TransformConfig,
    TransformStep,
    TransformationType
)
from aura_common.atomic.connectors import (
    KafkaProducer,
    KafkaMessage,
    KafkaConfig
)


@dataclass
class DataProcessingInput:
    """Input for data processing workflow."""
    
    data: Any
    data_type: str
    processing_rules: Dict[str, Any]
    output_topic: Optional[str] = None
    validation_rules: Optional[List[Dict[str, Any]]] = None
    transformation_steps: Optional[List[Dict[str, Any]]] = None


@dataclass
class DataProcessingResult:
    """Result of data processing workflow."""
    
    success: bool
    processed_data: Any
    validation_results: Optional[Dict[str, Any]] = None
    transformation_results: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    processing_time_ms: float = 0


class DataProcessingWorkflow:
    """
    Temporal workflow for data processing pipeline.
    
    Orchestrates validation, transformation, and enrichment
    of data using atomic components.
    """
    
    def __init__(self):
        self.workflow_id = None
        self.start_time = None
    
    async def run(self, input_data: DataProcessingInput) -> DataProcessingResult:
        """
        Execute data processing workflow.
        
        Args:
            input_data: Processing input parameters
            
        Returns:
            DataProcessingResult with processed data
        """
        errors = []
        processed_data = input_data.data
        
        try:
            # Step 1: Validation
            if input_data.validation_rules:
                validation_result = await DataValidationActivity.execute(
                    data=processed_data,
                    rules=input_data.validation_rules
                )
                
                if not validation_result["is_valid"]:
                    errors.extend(validation_result["errors"])
                    return DataProcessingResult(
                        success=False,
                        processed_data=None,
                        validation_results=validation_result,
                        errors=errors
                    )
            
            # Step 2: Transformation
            if input_data.transformation_steps:
                transformation_result = await DataTransformationActivity.execute(
                    data=processed_data,
                    steps=input_data.transformation_steps
                )
                
                processed_data = transformation_result["transformed_data"]
            
            # Step 3: Enrichment
            enrichment_result = await DataEnrichmentActivity.execute(
                data=processed_data,
                data_type=input_data.data_type,
                rules=input_data.processing_rules
            )
            
            processed_data = enrichment_result["enriched_data"]
            
            # Step 4: Output to Kafka if configured
            if input_data.output_topic:
                await self._publish_to_kafka(
                    processed_data,
                    input_data.output_topic
                )
            
            return DataProcessingResult(
                success=True,
                processed_data=processed_data,
                validation_results=validation_result if input_data.validation_rules else None,
                transformation_results=transformation_result if input_data.transformation_steps else None,
                errors=errors if errors else None
            )
            
        except Exception as e:
            errors.append(f"Workflow error: {str(e)}")
            return DataProcessingResult(
                success=False,
                processed_data=None,
                errors=errors
            )
    
    async def _publish_to_kafka(self, data: Any, topic: str):
        """Publish processed data to Kafka."""
        kafka_config = KafkaConfig(bootstrap_servers="localhost:9092")
        producer = KafkaProducer("workflow-producer", kafka_config)
        
        message = KafkaMessage(
            topic=topic,
            value=data,
            headers={"workflow_id": self.workflow_id}
        )
        
        await producer.process(message)


class DataValidationActivity:
    """Activity for data validation using atomic components."""
    
    @staticmethod
    async def execute(
        data: Any,
        rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute data validation activity.
        
        Args:
            data: Data to validate
            rules: Validation rules
            
        Returns:
            Validation results
        """
        # Convert rules to ValidationRule objects
        from aura_common.atomic.processors.validator import ValidationRule
        
        validation_rules = []
        for rule_dict in rules:
            validation_rules.append(
                ValidationRule(
                    name=rule_dict["name"],
                    validator=eval(rule_dict["validator"]),  # In prod, use safe eval
                    error_message=rule_dict["error_message"],
                    required=rule_dict.get("required", True)
                )
            )
        
        config = ValidationConfig(rules=validation_rules)
        validator = DataValidator("workflow-validator", config)
        
        result, metrics = await validator.process(data)
        
        return {
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "metadata": result.metadata,
            "execution_time_ms": metrics.execution_time_ms
        }


class DataTransformationActivity:
    """Activity for data transformation using atomic components."""
    
    @staticmethod
    async def execute(
        data: Any,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute data transformation activity.
        
        Args:
            data: Data to transform
            steps: Transformation steps
            
        Returns:
            Transformation results
        """
        # Convert steps to TransformStep objects
        transform_steps = []
        for step_dict in steps:
            transform_steps.append(
                TransformStep(
                    name=step_dict["name"],
                    transform_type=TransformationType[step_dict["type"]],
                    params=step_dict.get("params", {}),
                    on_error=step_dict.get("on_error", "fail")
                )
            )
        
        config = TransformConfig(steps=transform_steps)
        transformer = DataTransformer("workflow-transformer", config)
        
        result, metrics = await transformer.process(data)
        
        return {
            "transformed_data": result.transformed,
            "steps_applied": result.steps_applied,
            "lineage": result.lineage,
            "metadata": result.metadata,
            "execution_time_ms": metrics.execution_time_ms
        }


class DataEnrichmentActivity:
    """Activity for data enrichment based on type and rules."""
    
    @staticmethod
    async def execute(
        data: Any,
        data_type: str,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute data enrichment activity.
        
        Args:
            data: Data to enrich
            data_type: Type of data
            rules: Enrichment rules
            
        Returns:
            Enrichment results
        """
        enriched_data = data
        enrichments = {}
        
        # Text enrichment
        if data_type == "text" and isinstance(data, str):
            config = PreprocessorConfig(
                remove_html=rules.get("remove_html", True),
                lowercase=rules.get("lowercase", False)
            )
            preprocessor = TextPreprocessor("enrichment-preprocessor", config)
            
            result, _ = await preprocessor.process(data)
            
            enrichments["tokens"] = result.tokens
            enrichments["token_count"] = result.token_count
            enrichments["metadata"] = result.metadata
            enriched_data = result.cleaned
        
        # Structured data enrichment
        elif data_type == "structured" and isinstance(data, dict):
            # Add timestamps
            if rules.get("add_timestamp", True):
                from datetime import datetime
                enriched_data["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Add data quality scores
            if rules.get("add_quality_score", True):
                enrichments["quality_score"] = calculate_quality_score(data)
        
        return {
            "enriched_data": enriched_data,
            "enrichments": enrichments,
            "data_type": data_type
        }


def calculate_quality_score(data: Dict[str, Any]) -> float:
    """Calculate data quality score based on completeness and validity."""
    if not data:
        return 0.0
    
    # Simple quality score based on non-null fields
    total_fields = len(data)
    non_null_fields = sum(1 for v in data.values() if v is not None)
    
    return non_null_fields / total_fields if total_fields > 0 else 0.0