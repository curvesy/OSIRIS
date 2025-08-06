"""
Data validation atomic component.

This component provides configurable validation rules for various data types,
ensuring data quality and consistency in the processing pipeline.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import re
from datetime import datetime, timezone

from ..base import AtomicComponent
from ..base.exceptions import ValidationError


@dataclass
class ValidationRule:
    """Individual validation rule."""
    
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True
    
    def validate(self, value: Any) -> Optional[str]:
        """Run validation and return error message if failed."""
        if value is None and not self.required:
            return None
        
        try:
            if not self.validator(value):
                return self.error_message
        except Exception as e:
            return f"{self.error_message}: {str(e)}"
        
        return None


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    
    rules: List[ValidationRule] = field(default_factory=list)
    fail_fast: bool = True
    return_all_errors: bool = False
    max_errors: int = 10
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.max_errors <= 0:
            raise ValueError("max_errors must be positive")
        if not self.rules:
            raise ValueError("At least one validation rule required")


@dataclass
class ValidationResult:
    """Result of validation process."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    @property
    def error_count(self) -> int:
        """Get number of errors."""
        return len(self.errors)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self.warnings) > 0


class DataValidator(AtomicComponent[Any, ValidationResult, ValidationConfig]):
    """
    Atomic component for data validation.
    
    Supports configurable validation rules and comprehensive error reporting.
    """
    
    # Common validation patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    URL_PATTERN = re.compile(
        r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'
        r'([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)$'
    )
    
    def _validate_config(self) -> None:
        """Validate component configuration."""
        self.config.validate()
    
    async def _process(self, input_data: Any) -> ValidationResult:
        """
        Validate data against configured rules.
        
        Args:
            input_data: Data to validate
            
        Returns:
            ValidationResult with errors and metadata
        """
        errors = []
        warnings = []
        rules_checked = 0
        rules_passed = 0
        
        # Apply each rule
        for rule in self.config.rules:
            rules_checked += 1
            
            error = rule.validate(input_data)
            if error:
                errors.append(f"{rule.name}: {error}")
                
                # Stop if fail_fast is enabled
                if self.config.fail_fast:
                    break
                
                # Stop if max errors reached
                if len(errors) >= self.config.max_errors:
                    warnings.append(
                        f"Validation stopped after {self.config.max_errors} errors"
                    )
                    break
            else:
                rules_passed += 1
        
        # Build metadata
        metadata = {
            "rules_checked": rules_checked,
            "rules_passed": rules_passed,
            "rules_failed": rules_checked - rules_passed,
            "data_type": type(input_data).__name__,
            "validation_time": datetime.now(timezone.utc).isoformat()
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors if self.config.return_all_errors else errors[:1],
            warnings=warnings,
            metadata=metadata
        )
    
    @classmethod
    def create_string_validator(
        cls,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        **kwargs
    ) -> 'DataValidator':
        """Factory method for string validation."""
        rules = []
        
        if min_length is not None:
            rules.append(ValidationRule(
                name="min_length",
                validator=lambda x: isinstance(x, str) and len(x) >= min_length,
                error_message=f"String must be at least {min_length} characters"
            ))
        
        if max_length is not None:
            rules.append(ValidationRule(
                name="max_length",
                validator=lambda x: isinstance(x, str) and len(x) <= max_length,
                error_message=f"String must be at most {max_length} characters"
            ))
        
        if pattern:
            regex = re.compile(pattern)
            rules.append(ValidationRule(
                name="pattern",
                validator=lambda x: isinstance(x, str) and regex.match(x) is not None,
                error_message=f"String must match pattern: {pattern}"
            ))
        
        config = ValidationConfig(rules=rules, **kwargs)
        return cls("string_validator", config)
    
    @classmethod
    def create_email_validator(cls, **kwargs) -> 'DataValidator':
        """Factory method for email validation."""
        rule = ValidationRule(
            name="email_format",
            validator=lambda x: isinstance(x, str) and cls.EMAIL_PATTERN.match(x) is not None,
            error_message="Invalid email format"
        )
        
        config = ValidationConfig(rules=[rule], **kwargs)
        return cls("email_validator", config)