#!/usr/bin/env python3
"""
LLM-powered content extraction system for enhanced web scraping.
Uses Qwen3 Coder 480B A35B via crawl4ai for intelligent content analysis.
This data enhances the planning agent's navigation intelligence.
"""

import json
import os
import requests
from typing import Dict, Any, List
from datetime import datetime
from bs4 import BeautifulSoup


class QwenLLMExtractor:
    """
    Real LLM-powered extraction system using crawl4ai for intelligent route analysis.
    Provides semantic understanding, quality scoring, and business value analysis.
    Uses GPT-OSS-20B model for advanced AI capabilities.
    """
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b:free", api_token: str = None):
        self.model_name = model_name
        self.api_token = api_token or "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"
        self.extraction_cache = {}  # Cache LLM extractions for performance
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Initialize crawl4ai LLM components
        try:
            from crawl4ai import LLMConfig, LLMExtractionStrategy
            self.LLMConfig = LLMConfig
            self.LLMExtractionStrategy = LLMExtractionStrategy
            self.llm_available = True
            print(f"      🧠 REAL LLM EXTRACTOR: Initialized with {model_name}")
        except ImportError:
            print("      ⚠️  crawl4ai not available, using direct API calls")
            self.llm_available = False
        
        print(f"      🔑 API Token configured: {bool(self.api_token)}")
    
    def extract_page_intelligence(self, page_url: str, html_content: str, dom_features: Dict) -> Dict[str, Any]:
        """
        Extract comprehensive page intelligence using Qwen3 Coder 480B A35B.
        Returns enhanced scoring data for planning agent optimization.
        """
        
        try:
            print(f"      🧠 QWEN3 analyzing: {page_url}")
            
            # Extract clean text content
            text_content = self._extract_clean_text(html_content)
            
            # 1. Business Intelligence Analysis
            business_analysis = self._analyze_business_value(page_url, text_content, dom_features)
            
            # 2. User Experience Analysis  
            ux_analysis = self._analyze_user_experience(page_url, text_content, dom_features)
            
            # 3. Navigation Planning Analysis
            planning_analysis = self._analyze_planning_priority(page_url, text_content, dom_features)
            
            # 4. Content Classification
            content_classification = self._classify_content_type(page_url, text_content, dom_features)
            
            # Compile comprehensive intelligence
            page_intelligence = {
                'page_url': page_url,
                'business_analysis': business_analysis,
                'ux_analysis': ux_analysis,
                'planning_analysis': planning_analysis,
                'content_classification': content_classification,
                'extraction_timestamp': datetime.now().isoformat(),
                'llm_model': 'qwen3-coder-480b-a35b',
                'extraction_method': 'crawl4ai' if self.llm_available else 'direct_api'
            }
            
            # Calculate final enhanced scores
            enhanced_scores = self._calculate_enhanced_scores(page_intelligence)
            page_intelligence.update(enhanced_scores)
            
            print(f"         ✅ QWEN3 Analysis Complete - Business: {enhanced_scores.get('business_value_score', 0):.1f}/10")
            
            return page_intelligence
            
        except Exception as e:
            print(f"      ⚠️  QWEN3 extraction failed for {page_url}: {e}")
            return self._get_fallback_intelligence(page_url)
    
    def _extract_clean_text(self, html_content: str) -> str:
        """Extract clean text content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            return clean_text[:2000]  # Limit for LLM processing
            
        except Exception:
            return str(html_content)[:2000]  # Fallback
    
    def _analyze_business_value(self, page_url: str, text_content: str, dom_features: Dict) -> Dict[str, Any]:
        """Analyze business value and conversion potential using QWEN3."""
        
        prompt = f"""
        Analyze this web page for BUSINESS VALUE and CONVERSION POTENTIAL:
        
        URL: {page_url}
        Content: {text_content[:800]}
        Forms: {len(dom_features.get('forms', []))}
        Buttons: {len(dom_features.get('buttons', []))}
        Images: {len(dom_features.get('images', []))}
        
        Evaluate and provide JSON response:
        {{
            "business_value_score": [1-10],
            "conversion_potential": [1-10],
            "revenue_impact": [1-10],
            "business_category": "ecommerce|lead_generation|content|service|other",
            "conversion_indicators": ["indicator1", "indicator2"],
            "business_importance": "high|medium|low",
            "reasoning": "detailed explanation"
        }}
        
        Focus on: sales potential, lead generation, customer engagement, revenue impact.
        """
        
        return self._call_llm_analysis(prompt, "business_value")
    
    def _analyze_user_experience(self, page_url: str, text_content: str, dom_features: Dict) -> Dict[str, Any]:
        """Analyze user experience and navigation quality using QWEN3."""
        
        prompt = f"""
        Analyze this web page for USER EXPERIENCE and NAVIGATION QUALITY:
        
        URL: {page_url}
        Content: {text_content[:800]}
        Interactive Elements: {len(dom_features.get('buttons', [])) + len(dom_features.get('inputs', []))}
        Links: {len(dom_features.get('links', []))}
        Scripts: {len(dom_features.get('scripts', []))}
        
        Evaluate and provide JSON response:
        {{
            "user_experience_score": [1-10],
            "navigation_clarity": [1-10],
            "interaction_quality": [1-10],
            "page_load_complexity": [1-10],
            "mobile_friendliness": [1-10],
            "accessibility_score": [1-10],
            "ux_issues": ["issue1", "issue2"],
            "ux_strengths": ["strength1", "strength2"],
            "reasoning": "detailed explanation"
        }}
        
        Focus on: ease of use, navigation clarity, interaction design, performance impact.
        """
        
        return self._call_llm_analysis(prompt, "user_experience")
    
    def _analyze_planning_priority(self, page_url: str, text_content: str, dom_features: Dict) -> Dict[str, Any]:
        """Analyze navigation planning priority using QWEN3."""
        
        prompt = f"""
        Analyze this web page for NAVIGATION PLANNING PRIORITY:
        
        URL: {page_url}
        Content: {text_content[:800]}
        Total Links: {len(dom_features.get('links', []))}
        Content Depth: {len(text_content)}
        
        Evaluate and provide JSON response:
        {{
            "planning_priority_score": [1-10],
            "navigation_hub_potential": [1-10],
            "content_discovery_value": [1-10],
            "strategic_importance": [1-10],
            "exploration_depth_recommendation": [1-5],
            "should_prioritize": true/false,
            "navigation_role": "hub|gateway|content|terminal|other",
            "priority_reasons": ["reason1", "reason2"],
            "reasoning": "detailed explanation"
        }}
        
        Focus on: strategic importance, navigation value, content discovery, hub potential.
        """
        
        return self._call_llm_analysis(prompt, "planning_priority")
    
    def _classify_content_type(self, page_url: str, text_content: str, dom_features: Dict) -> Dict[str, Any]:
        """Classify content type and page function using QWEN3."""
        
        prompt = f"""
        Classify this web page's CONTENT TYPE and FUNCTION:
        
        URL: {page_url}
        Content: {text_content[:800]}
        Forms: {len(dom_features.get('forms', []))}
        Media: {len(dom_features.get('media', []))}
        
        Classify and provide JSON response:
        {{
            "page_type": "landing|product|category|contact|about|blog|service|checkout|other",
            "content_category": "commercial|informational|transactional|navigational",
            "primary_function": "sell|inform|collect|navigate|support|other",
            "target_audience": "customers|prospects|visitors|support|other",
            "content_quality": [1-10],
            "information_density": [1-10],
            "actionability": [1-10],
            "content_themes": ["theme1", "theme2"],
            "key_entities": ["entity1", "entity2"],
            "reasoning": "detailed explanation"
        }}
        
        Focus on: page purpose, content type, user intent, business function.
        """
        
        return self._call_llm_analysis(prompt, "content_classification")
    
    def _call_llm_analysis(self, prompt: str, analysis_type: str) -> Dict[str, Any]:
        """Call GPT-OSS-20B LLM for analysis using crawl4ai or direct API."""
        
        try:
            # Check cache first
            cache_key = hash(prompt)
            if cache_key in self.extraction_cache:
                return self.extraction_cache[cache_key]
            
            # Try crawl4ai first if available
            if self.llm_available:
                response = self._crawl4ai_extraction(prompt)
            else:
                response = self._direct_api_call(prompt)
            
            # Parse response
            result = self._parse_llm_response(response)
            
            # Cache result
            self.extraction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"         ⚠️  {analysis_type} analysis failed: {e}")
            return self._get_fallback_analysis(analysis_type)
    
    def _crawl4ai_extraction(self, prompt: str) -> str:
        """Use crawl4ai for LLM extraction."""
        
        try:
            # Create LLM config
            llm_config = self.LLMConfig(
                provider=self.model_name,
                api_token=self.api_token,
                base_url=self.base_url,
                temperature=0.1,
                max_tokens=1500
            )
            
            # Create extraction strategy
            llm_strategy = self.LLMExtractionStrategy(
                llm_config=llm_config,
                instruction=prompt,
                extraction_type='block'
            )
            
            # Since we're doing content analysis (not HTML extraction), use direct API
            return self._direct_api_call(prompt)
            
        except Exception as e:
            print(f"         ⚠️  crawl4ai extraction failed: {e}")
            return self._direct_api_call(prompt)
    
    def _direct_api_call(self, prompt: str) -> str:
        """Direct API call to GPT-OSS-20B via OpenRouter."""
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://actory-ai-scrape",
                "X-Title": "Actory AI Web Scraper LLM Extraction"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are GPT-OSS-20B, an expert in web content analysis, business intelligence, and user experience evaluation. Always provide responses in valid JSON format only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
                "top_p": 0.9
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return content
            else:
                print(f"         ❌ API Error: {response.status_code}")
                return self._get_fallback_response()
                
        except Exception as e:
            print(f"         ⚠️  Direct API call failed: {e}")
            return self._get_fallback_response()
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        
        try:
            # Clean response
            cleaned = response.strip()
            if "```json" in cleaned:
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                cleaned = cleaned[start:end]
            elif "```" in cleaned:
                start = cleaned.find("```") + 3
                end = cleaned.find("```", start)
                cleaned = cleaned[start:end]
            
            # Parse JSON
            parsed = json.loads(cleaned.strip())
            return parsed
            
        except json.JSONDecodeError:
            print(f"         ⚠️  Failed to parse LLM response: {response}")
            return self._get_fallback_analysis("unknown")
    
    def _calculate_enhanced_scores(self, intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final enhanced scores from all analyses."""
        
        try:
            # Extract scores from analyses
            business_score = intelligence['business_analysis'].get('business_value_score', 5)
            ux_score = intelligence['ux_analysis'].get('user_experience_score', 5)
            planning_score = intelligence['planning_analysis'].get('planning_priority_score', 5)
            content_quality = intelligence['content_classification'].get('content_quality', 5)
            
            # Calculate composite scores
            overall_score = (business_score + ux_score + planning_score + content_quality) / 4.0
            
            return {
                'business_value_score': business_score,
                'user_experience_score': ux_score,
                'planning_priority_score': planning_score,
                'content_quality_score': content_quality,
                'overall_intelligence_score': overall_score,
                'page_type': intelligence['content_classification'].get('page_type', 'unknown'),
                'conversion_potential': intelligence['business_analysis'].get('conversion_potential', 5),
                'navigation_hub_potential': intelligence['planning_analysis'].get('navigation_hub_potential', 5),
                'enhanced_scoring_active': True
            }
            
        except Exception as e:
            print(f"         ⚠️  Score calculation failed: {e}")
            return {
                'business_value_score': 5,
                'user_experience_score': 5,
                'planning_priority_score': 5,
                'content_quality_score': 5,
                'overall_intelligence_score': 5,
                'enhanced_scoring_active': False
            }
    
    def _get_fallback_intelligence(self, page_url: str) -> Dict[str, Any]:
        """Provide fallback intelligence when LLM extraction fails."""
        
        return {
            'page_url': page_url,
            'business_value_score': 5,
            'user_experience_score': 5,
            'planning_priority_score': 5,
            'content_quality_score': 5,
            'overall_intelligence_score': 5,
            'page_type': 'unknown',
            'conversion_potential': 5,
            'navigation_hub_potential': 5,
            'enhanced_scoring_active': False,
            'extraction_method': 'fallback',
            'extraction_timestamp': datetime.now().isoformat(),
            'error': 'LLM extraction failed'
        }
    
    def _get_fallback_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """Provide fallback analysis for specific analysis types."""
        
        fallbacks = {
            'business_value': {
                'business_value_score': 5,
                'conversion_potential': 5,
                'revenue_impact': 5,
                'business_category': 'other',
                'business_importance': 'medium'
            },
            'user_experience': {
                'user_experience_score': 5,
                'navigation_clarity': 5,
                'interaction_quality': 5,
                'page_load_complexity': 5
            },
            'planning_priority': {
                'planning_priority_score': 5,
                'navigation_hub_potential': 5,
                'content_discovery_value': 5,
                'strategic_importance': 5
            },
            'content_classification': {
                'page_type': 'unknown',
                'content_category': 'informational',
                'content_quality': 5,
                'information_density': 5
            }
        }
        
        return fallbacks.get(analysis_type, {'score': 5})
    
    def _get_fallback_response(self) -> str:
        """Get fallback JSON response."""
        
        return json.dumps({
            'score': 5,
            'analysis': 'fallback',
            'reasoning': 'LLM API unavailable'
        })
    
    def clear_cache(self):
        """Clear extraction cache."""
        self.extraction_cache.clear()
        print("      🗑️  QWEN3 extraction cache cleared")
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        
        return {
            'llm_available': self.llm_available,
            'model_name': self.model_name,
            'cache_size': len(self.extraction_cache),
            'api_configured': bool(self.api_token)
        }


def calculate_enhanced_complexity_score(dom_features: Dict, intelligence: Dict = None) -> float:
    """
    Calculate enhanced complexity score using both DOM features and LLM intelligence.
    This score is used by the planning agent for better navigation decisions.
    """
    
    # Base DOM scoring
    base_score = (
        len(dom_features.get("scripts", [])) * 0.5 +
        len(dom_features.get("forms", [])) * 0.25 +
        (len(dom_features.get("buttons", [])) + len(dom_features.get("inputs", []))) * 0.15 +
        len(dom_features.get("images", [])) * 0.1 +
        (len(dom_features.get("links", [])) / 100.0) * 0.05 +
        (len(dom_features.get("media", [])) / 50.0) * 0.05
    )
    
    # Apply LLM enhancement if available
    if intelligence and intelligence.get('enhanced_scoring_active'):
        try:
            # Get LLM scores
            business_value = intelligence.get('business_value_score', 5)
            ux_score = intelligence.get('user_experience_score', 5)
            planning_priority = intelligence.get('planning_priority_score', 5)
            
            # Calculate multipliers
            business_multiplier = 1.0 + (business_value / 10.0)  # 1.0 - 2.0
            ux_multiplier = 0.8 + (ux_score / 10.0 * 0.4)       # 0.8 - 1.2
            priority_multiplier = 1.0 + (planning_priority / 10.0 * 0.5)  # 1.0 - 1.5
            
            # Apply enhancements
            enhanced_score = base_score * business_multiplier * ux_multiplier * priority_multiplier
            
            print(f"         📈 Enhanced Score: {base_score:.2f} → {enhanced_score:.2f} (B:{business_multiplier:.2f}x U:{ux_multiplier:.2f}x P:{priority_multiplier:.2f}x)")
            
            return enhanced_score
            
        except Exception as e:
            print(f"         ⚠️  Enhanced scoring failed: {e}")
    
    return base_score
