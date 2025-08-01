"""
Text preprocessing atomic component.

This component handles text cleaning, normalization, and tokenization
as a single-responsibility unit in the processing pipeline.
"""

from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
import unicodedata

from ..base import AtomicComponent
from ..base.exceptions import ValidationError


@dataclass
class PreprocessorConfig:
    """Configuration for text preprocessing."""
    
    max_length: int = 10000
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = False
    lowercase: bool = False
    remove_extra_whitespace: bool = True
    remove_special_chars: bool = False
    remove_numbers: bool = False
    normalize_unicode: bool = True
    min_token_length: int = 1
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.min_token_length < 0:
            raise ValueError("min_token_length must be non-negative")


@dataclass
class ProcessedText:
    """Result of text preprocessing."""
    
    original: str
    cleaned: str
    tokens: List[str]
    metadata: Dict[str, Any]
    
    @property
    def token_count(self) -> int:
        """Get number of tokens."""
        return len(self.tokens)
    
    @property
    def char_count(self) -> int:
        """Get character count of cleaned text."""
        return len(self.cleaned)


class TextPreprocessor(AtomicComponent[str, ProcessedText, PreprocessorConfig]):
    """
    Atomic component for text preprocessing.
    
    Handles cleaning, normalization, and tokenization of text data.
    """
    
    # Regex patterns compiled once for efficiency
    HTML_PATTERN = re.compile(r'<[^>]+>')
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    EMAIL_PATTERN = re.compile(r'\S+@\S+')
    MULTI_SPACE_PATTERN = re.compile(r'\s+')
    
    def _validate_config(self) -> None:
        """Validate component configuration."""
        self.config.validate()
    
    async def _process(self, input_data: str) -> ProcessedText:
        """
        Process text through cleaning pipeline.
        
        Args:
            input_data: Raw text to process
            
        Returns:
            ProcessedText with cleaned text and metadata
        """
        # Handle empty input
        if not input_data:
            return ProcessedText(
                original="",
                cleaned="",
                tokens=[],
                metadata={"empty_input": True}
            )
        
        # Track original length
        original_length = len(input_data)
        
        # Truncate if needed
        text = input_data[:self.config.max_length]
        truncated = len(input_data) > self.config.max_length
        
        # Store original for reference
        original = text
        
        # Apply cleaning pipeline
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        if self.config.remove_html:
            text = self.HTML_PATTERN.sub(' ', text)
        
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)
        
        if self.config.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)
        
        if self.config.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        if self.config.remove_numbers:
            text = re.sub(r'\d+', ' ', text)
        
        if self.config.remove_extra_whitespace:
            text = self.MULTI_SPACE_PATTERN.sub(' ', text).strip()
        
        if self.config.lowercase:
            text = text.lower()
        
        # Tokenize
        tokens = self._tokenize(text)
        
        # Filter tokens by length
        if self.config.min_token_length > 0:
            tokens = [t for t in tokens if len(t) >= self.config.min_token_length]
        
        # Build metadata
        metadata = {
            "original_length": original_length,
            "cleaned_length": len(text),
            "token_count": len(tokens),
            "truncated": truncated,
            "avg_token_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            "unique_tokens": len(set(tokens))
        }
        
        return ProcessedText(
            original=original,
            cleaned=text,
            tokens=tokens,
            metadata=metadata
        )
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFKD form and remove non-ASCII
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.split() if text else []
    
    async def validate_input(self, data: str) -> bool:
        """Validate input before processing."""
        if not isinstance(data, str):
            raise ValidationError(
                f"Expected string input, got {type(data).__name__}",
                component_name=self.name
            )
        return True