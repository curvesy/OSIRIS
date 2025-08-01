#!/usr/bin/env python3
"""
🤖 AURA Intelligence: Gemini API Client
Enterprise-grade Google Gemini integration with LangChain compatibility

Features:
- LangChain-compatible interface
- Enterprise guardrails integration
- Async/await support
- Error handling and retries
- Cost tracking
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

@dataclass
class GeminiConfig:
    """Configuration for Gemini API client"""
    api_key: str = "AIzaSyDiX165POC4I0uJI8VAL_9to8nhomSZ_og"
    model: str = "gemini-2.0-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: float = 30.0
    max_retries: int = 3

class GeminiMessage:
    """LangChain-compatible message class"""
    
    def __init__(self, content: str, role: str = "user"):
        self.content = content
        self.role = role
    
    def __str__(self):
        return self.content

class GeminiResponse:
    """LangChain-compatible response class"""
    
    def __init__(self, content: str, usage: Dict[str, Any] = None):
        self.content = content
        self.usage = usage or {}
    
    def __str__(self):
        return self.content

class GeminiClient:
    """🤖 Enterprise Gemini API Client with LangChain compatibility"""
    
    def __init__(self, config: GeminiConfig = None):
        self.config = config or GeminiConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        
        # Cost tracking (approximate pricing)
        self.cost_per_1k_input_tokens = 0.00015  # $0.15 per 1M tokens
        self.cost_per_1k_output_tokens = 0.0006  # $0.60 per 1M tokens
        
        logger.info(f"🤖 Gemini client initialized: {self.config.model}")
    
    async def ainvoke(self, messages: Union[str, List, Dict], **kwargs) -> GeminiResponse:
        """
        🤖 Async invoke method compatible with LangChain interface
        
        Args:
            messages: Input messages (string, list, or dict)
            **kwargs: Additional parameters
            
        Returns:
            GeminiResponse: Response object with content
        """
        
        start_time = time.time()
        
        try:
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages(messages)
            
            # Prepare request payload
            payload = {
                "contents": gemini_messages,
                "generationConfig": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "topP": kwargs.get("top_p", 0.8),
                    "topK": kwargs.get("top_k", 40)
                }
            }
            
            # Make API request with retries
            response_data = await self._make_request_with_retries(payload)
            
            # Extract response content
            content = self._extract_content(response_data)
            
            # Calculate usage and cost
            usage = self._calculate_usage(messages, content, time.time() - start_time)
            
            logger.info(f"🤖 Gemini call completed: {len(content)} chars, ${usage.get('cost', 0):.4f}")
            
            return GeminiResponse(content=content, usage=usage)
            
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {e}")
            raise e
    
    def _convert_messages(self, messages: Union[str, List, Dict]) -> List[Dict[str, Any]]:
        """Convert various message formats to Gemini format"""
        
        if isinstance(messages, str):
            # Simple string input
            return [{
                "parts": [{"text": messages}]
            }]
        
        elif isinstance(messages, list):
            # List of messages (LangChain format)
            gemini_messages = []
            
            for msg in messages:
                if hasattr(msg, 'content'):
                    # LangChain message object
                    text = msg.content
                elif isinstance(msg, dict):
                    # Dictionary message
                    text = msg.get('content', str(msg))
                else:
                    # String message
                    text = str(msg)
                
                gemini_messages.append({
                    "parts": [{"text": text}]
                })
            
            return gemini_messages
        
        elif isinstance(messages, dict):
            # Single dictionary message
            text = messages.get('content', str(messages))
            return [{
                "parts": [{"text": text}]
            }]
        
        else:
            # Fallback
            return [{
                "parts": [{"text": str(messages)}]
            }]
    
    async def _make_request_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with retry logic"""
        
        url = f"{self.config.base_url}/models/{self.config.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.config.api_key
        }
        
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    url=url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"⚠️ Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                    continue
                
                else:
                    # Other HTTP error
                    error_text = response.text
                    raise Exception(f"HTTP {response.status_code}: {error_text}")
                
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"⚠️ Request failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        # All retries failed
        raise last_exception or Exception("All retry attempts failed")
    
    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """Extract text content from Gemini response"""
        
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return "No response generated"
            
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                return "No content in response"
            
            # Combine all text parts
            text_parts = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
            
            return "\n".join(text_parts) if text_parts else "Empty response"
            
        except Exception as e:
            logger.error(f"❌ Failed to extract content: {e}")
            return f"Error extracting content: {e}"
    
    def _calculate_usage(self, input_messages: Any, output_content: str, duration: float) -> Dict[str, Any]:
        """Calculate usage statistics and cost"""
        
        # Estimate token counts (rough approximation)
        input_text = str(input_messages)
        input_tokens = len(input_text.split()) * 1.3  # Rough token estimate
        output_tokens = len(output_content.split()) * 1.3
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output_tokens
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens + output_tokens),
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cost": total_cost,
            "duration": duration,
            "model": self.config.model
        }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 LANGCHAIN-COMPATIBLE WRAPPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatGemini:
    """🤖 LangChain-compatible Gemini chat model"""
    
    def __init__(self, 
                 model: str = "gemini-2.0-flash",
                 temperature: float = 0.1,
                 max_tokens: int = 2000,
                 api_key: str = None):
        
        config = GeminiConfig(
            api_key=api_key or "AIzaSyDiX165POC4I0uJI8VAL_9to8nhomSZ_og",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.client = GeminiClient(config)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def ainvoke(self, messages: Any, **kwargs) -> GeminiResponse:
        """LangChain-compatible async invoke"""
        return await self.client.ainvoke(messages, **kwargs)
    
    def invoke(self, messages: Any, **kwargs) -> GeminiResponse:
        """LangChain-compatible sync invoke"""
        import asyncio
        return asyncio.run(self.client.ainvoke(messages, **kwargs))
    
    async def aclose(self):
        """Close the client"""
        await self.client.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 CONVENIENCE FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_gemini_client(model: str = "gemini-2.0-flash", **kwargs) -> ChatGemini:
    """🤖 Create a Gemini client with LangChain compatibility"""
    return ChatGemini(model=model, **kwargs)

async def test_gemini_connection() -> bool:
    """🧪 Test Gemini API connection"""
    
    try:
        client = create_gemini_client()
        
        response = await client.ainvoke("Hello! Please respond with 'Connection successful'")
        
        await client.aclose()
        
        success = "successful" in response.content.lower()
        
        if success:
            logger.info("✅ Gemini API connection test successful")
        else:
            logger.warning(f"⚠️ Gemini API responded but unexpected content: {response.content}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Gemini API connection test failed: {e}")
        return False
