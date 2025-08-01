"""
Data transformation atomic component.

This component provides configurable data transformations with support
for chaining, conditional logic, and type safety.
"""

from typing import Any, Dict, List, Optional, Callable, Union, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from ..base import AtomicComponent
from ..base.exceptions import ProcessingError, ValidationError

T = TypeVar('T')


class TransformationType(Enum):
    """Supported transformation types."""
    
    JSON_PARSE = "json_parse"
    JSON_STRINGIFY = "json_stringify"
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    TRIM = "trim"
    REPLACE = "replace"
    EXTRACT = "extract"
    FLATTEN = "flatten"
    NEST = "nest"
    CUSTOM = "custom"


@dataclass
class TransformStep:
    """Individual transformation step."""
    
    name: str
    transform_type: TransformationType
    params: Dict[str, Any] = field(default_factory=dict)
    custom_fn: Optional[Callable[[Any], Any]] = None
    on_error: str = "fail"  # "fail", "skip", "default"
    default_value: Any = None
    
    def apply(self, data: Any) -> Any:
        """Apply transformation to data."""
        try:
            if self.transform_type == TransformationType.CUSTOM:
                if not self.custom_fn:
                    raise ValueError("Custom function required for CUSTOM type")
                return self.custom_fn(data)
            
            # Built-in transformations
            if self.transform_type == TransformationType.JSON_PARSE:
                return json.loads(data) if isinstance(data, str) else data
            
            elif self.transform_type == TransformationType.JSON_STRINGIFY:
                return json.dumps(data)
            
            elif self.transform_type == TransformationType.UPPERCASE:
                return data.upper() if isinstance(data, str) else str(data).upper()
            
            elif self.transform_type == TransformationType.LOWERCASE:
                return data.lower() if isinstance(data, str) else str(data).lower()
            
            elif self.transform_type == TransformationType.TRIM:
                return data.strip() if isinstance(data, str) else str(data).strip()
            
            elif self.transform_type == TransformationType.REPLACE:
                old = self.params.get("old", "")
                new = self.params.get("new", "")
                return data.replace(old, new) if isinstance(data, str) else str(data).replace(old, new)
            
            elif self.transform_type == TransformationType.EXTRACT:
                key = self.params.get("key")
                if isinstance(data, dict) and key:
                    return data.get(key, self.default_value)
                return self.default_value
            
            elif self.transform_type == TransformationType.FLATTEN:
                if isinstance(data, dict):
                    return self._flatten_dict(data)
                return data
            
            elif self.transform_type == TransformationType.NEST:
                key = self.params.get("key", "data")
                return {key: data}
            
            else:
                raise ValueError(f"Unknown transformation type: {self.transform_type}")
                
        except Exception as e:
            if self.on_error == "fail":
                raise
            elif self.on_error == "skip":
                return data
            else:  # default
                return self.default_value
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


@dataclass
class TransformConfig:
    """Configuration for data transformation."""
    
    steps: List[TransformStep] = field(default_factory=list)
    validate_output: bool = True
    track_lineage: bool = True
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.steps:
            raise ValueError("At least one transformation step required")
        
        for step in self.steps:
            if step.transform_type == TransformationType.CUSTOM and not step.custom_fn:
                raise ValueError(f"Step '{step.name}' requires custom function")


@dataclass
class TransformResult:
    """Result of transformation process."""
    
    original: Any
    transformed: Any
    steps_applied: List[str]
    lineage: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DataTransformer(AtomicComponent[Any, TransformResult, TransformConfig]):
    """
    Atomic component for data transformation.
    
    Supports chaining multiple transformations with error handling.
    """
    
    def _validate_config(self) -> None:
        """Validate component configuration."""
        self.config.validate()
    
    async def _process(self, input_data: Any) -> TransformResult:
        """
        Apply transformation pipeline to data.
        
        Args:
            input_data: Data to transform
            
        Returns:
            TransformResult with transformed data and lineage
        """
        current_data = input_data
        steps_applied = []
        lineage = []
        
        # Track start time
        start_time = datetime.utcnow()
        
        # Apply each transformation step
        for step in self.config.steps:
            step_start = datetime.utcnow()
            
            # Store pre-transform state for lineage
            if self.config.track_lineage:
                lineage.append({
                    "step": step.name,
                    "type": step.transform_type.value,
                    "input": self._safe_repr(current_data),
                    "timestamp": step_start.isoformat()
                })
            
            try:
                # Apply transformation
                transformed = step.apply(current_data)
                current_data = transformed
                steps_applied.append(step.name)
                
                # Update lineage with output
                if self.config.track_lineage and lineage:
                    lineage[-1]["output"] = self._safe_repr(current_data)
                    lineage[-1]["duration_ms"] = (
                        datetime.utcnow() - step_start
                    ).total_seconds() * 1000
                    
            except Exception as e:
                self.logger.warning(
                    f"Transformation step '{step.name}' failed",
                    error=str(e),
                    step_type=step.transform_type.value
                )
                
                if step.on_error == "fail":
                    raise ProcessingError(
                        f"Transformation failed at step '{step.name}': {str(e)}",
                        component_name=self.name
                    )
        
        # Validate output if configured
        if self.config.validate_output:
            self._validate_output(current_data)
        
        # Build metadata
        metadata = {
            "total_steps": len(self.config.steps),
            "steps_applied": len(steps_applied),
            "steps_skipped": len(self.config.steps) - len(steps_applied),
            "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            "input_type": type(input_data).__name__,
            "output_type": type(current_data).__name__
        }
        
        return TransformResult(
            original=input_data,
            transformed=current_data,
            steps_applied=steps_applied,
            lineage=lineage if self.config.track_lineage else [],
            metadata=metadata
        )
    
    def _safe_repr(self, data: Any, max_length: int = 100) -> str:
        """Get safe string representation of data."""
        try:
            repr_str = repr(data)
            if len(repr_str) > max_length:
                return repr_str[:max_length] + "..."
            return repr_str
        except:
            return f"<{type(data).__name__}>"
    
    def _validate_output(self, data: Any) -> None:
        """Validate transformation output."""
        if data is None:
            self.logger.warning("Transformation resulted in None value")
    
    @classmethod
    def create_json_transformer(cls, **kwargs) -> 'DataTransformer':
        """Factory for JSON parsing transformer."""
        step = TransformStep(
            name="json_parse",
            transform_type=TransformationType.JSON_PARSE
        )
        config = TransformConfig(steps=[step], **kwargs)
        return cls("json_transformer", config)