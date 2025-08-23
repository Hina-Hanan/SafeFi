"""
Social Sentiment Analyzer for SafeFi DeFi Risk Assessment Agent.

This module provides social sentiment analysis capabilities for DeFi protocols
using social media data and news sentiment scoring.
"""

import asyncio
from typing import Dict, Any, List, Optional
import aiohttp
from datetime import datetime, timedelta
from loguru import logger
import numpy as np

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available for sentiment analysis")

from ..utils.logger import get_logger, log_function_call, log_error_with_context
from ..config.settings import get_settings


class SentimentAnalyzer:
    """
    Social sentiment analysis for DeFi protocols.
    
    Analyzes social media mentions, news articles, and community discussions
    to generate sentiment scores for risk assessment.
    """
    
    def __init__(self):
        """Initialize SentimentAnalyzer."""
        self.settings = get_settings()
        self.logger = get_logger("SentimentAnalyzer")
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Sentiment keywords for DeFi-specific analysis
        self.positive_keywords = [
            'bullish', 'moon', 'pump', 'gem', 'amazing', 'great', 'love',
            'solid', 'strong', 'buying', 'hodl', 'excellent', 'fantastic'
        ]
        
        self.negative_keywords = [
            'bearish', 'dump', 'crash', 'scam', 'rug', 'avoid', 'warning',
            'risky', 'dangerous', 'selling', 'exit', 'terrible', 'awful'
        ]
        
        self.risk_keywords = [
            'exploit', 'hack', 'vulnerability', 'audit', 'bug', 'attack',
            'suspicious', 'centralized', 'manipulation', 'whale'
        ]
    
    async def analyze_protocol_sentiment(self, protocol_name: str, 
                                       days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze sentiment for a specific DeFi protocol.
        
        Args:
            protocol_name: Name of the protocol
            days_back: Number of days to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        log_function_call("SentimentAnalyzer.analyze_protocol_sentiment", {
            "protocol_name": protocol_name,
            "days_back": days_back
        })
        
        try:
            # Collect social media mentions (mock data for demo)
            mentions = await self._collect_social_mentions(protocol_name, days_back)
            
            if not mentions:
                return self._get_neutral_sentiment_result(protocol_name)
            
            # Analyze sentiment of collected mentions
            sentiment_results = await self._analyze_sentiment_batch(mentions)
            
            # Calculate aggregated sentiment scores
            aggregated_sentiment = self._aggregate_sentiment_scores(sentiment_results)
            
            # Add protocol-specific context
            aggregated_sentiment.update({
                'protocol_name': protocol_name,
                'analysis_period_days': days_back,
                'mentions_analyzed': len(mentions),
                'analysis_timestamp': datetime.utcnow().isoformat()
            })
            
            return aggregated_sentiment
            
        except Exception as e:
            log_error_with_context(e, {"protocol_name": protocol_name, "days_back": days_back})
            return self._get_neutral_sentiment_result(protocol_name, error=str(e))
    
    async def _collect_social_mentions(self, protocol_name: str, 
                                     days_back: int) -> List[Dict[str, Any]]:
        """
        Collect social media mentions for a protocol.
        
        Args:
            protocol_name: Protocol name to search for
            days_back: Days to look back
            
        Returns:
            List of social media mentions
        """
        try:
            # Mock social media data for demonstration
            # In production, integrate with Twitter API, Reddit API, etc.
            mock_mentions = [
                {
                    'text': f'{protocol_name} is looking really strong this week!',
                    'source': 'twitter',
                    'timestamp': datetime.utcnow() - timedelta(days=1),
                    'engagement': 45
                },
                {
                    'text': f'Concerned about {protocol_name} centralization issues',
                    'source': 'reddit',
                    'timestamp': datetime.utcnow() - timedelta(days=2),
                    'engagement': 23
                },
                {
                    'text': f'{protocol_name} TVL growing steadily, good fundamentals',
                    'source': 'telegram',
                    'timestamp': datetime.utcnow() - timedelta(days=3),
                    'engagement': 67
                },
                {
                    'text': f'Recent {protocol_name} update improved security significantly',
                    'source': 'discord',
                    'timestamp': datetime.utcnow() - timedelta(days=4),
                    'engagement': 89
                },
                {
                    'text': f'Warning: {protocol_name} showing high volatility patterns',
                    'source': 'twitter',
                    'timestamp': datetime.utcnow() - timedelta(days=5),
                    'engagement': 156
                }
            ]
            
            self.logger.debug(f"Collected {len(mock_mentions)} mentions for {protocol_name}")
            return mock_mentions
            
        except Exception as e:
            log_error_with_context(e, {"protocol_name": protocol_name})
            return []
    
    async def _analyze_sentiment_batch(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of mentions.
        
        Args:
            mentions: List of social media mentions
            
        Returns:
            List of sentiment analysis results
        """
        try:
            results = []
            
            for mention in mentions:
                sentiment_result = await self._analyze_single_mention(mention)
                results.append(sentiment_result)
            
            return results
            
        except Exception as e:
            log_error_with_context(e, {"mentions_count": len(mentions)})
            return []
    
    async def _analyze_single_mention(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment for a single mention.
        
        Args:
            mention: Single social media mention
            
        Returns:
            Sentiment analysis result
        """
        try:
            text = mention.get('text', '')
            
            if TEXTBLOB_AVAILABLE:
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1
                
                # Convert to 0-1 scale
                sentiment_score = (polarity + 1) / 2
                
            else:
                # Fallback to keyword-based sentiment analysis
                sentiment_score = self._keyword_based_sentiment(text)
                subjectivity = 0.5  # Default
            
            # Adjust for DeFi-specific risk keywords
            risk_adjustment = self._calculate_risk_adjustment(text)
            
            # Weight by engagement if available
            engagement_weight = min(mention.get('engagement', 1) / 100, 2.0)  # Cap at 2x
            
            # Final weighted sentiment score
            weighted_sentiment = sentiment_score * engagement_weight + risk_adjustment
            weighted_sentiment = max(0, min(1, weighted_sentiment))  # Clamp to [0,1]
            
            return {
                'text': text,
                'source': mention.get('source', 'unknown'),
                'raw_sentiment': float(sentiment_score),
                'weighted_sentiment': float(weighted_sentiment),
                'subjectivity': float(subjectivity),
                'engagement': mention.get('engagement', 0),
                'risk_adjustment': float(risk_adjustment),
                'timestamp': mention.get('timestamp', datetime.utcnow()).isoformat()
            }
            
        except Exception as e:
            log_error_with_context(e, {"mention": mention})
            return {
                'text': mention.get('text', ''),
                'sentiment': 0.5,  # Neutral fallback
                'error': str(e)
            }
    
    def _keyword_based_sentiment(self, text: str) -> float:
        """
        Simple keyword-based sentiment analysis fallback.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (0-1 scale)
        """
        try:
            text_lower = text.lower()
            
            positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
            negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
            
            if positive_count == 0 and negative_count == 0:
                return 0.5  # Neutral
            
            # Calculate sentiment based on keyword counts
            sentiment = (positive_count - negative_count) / (positive_count + negative_count + 1)
            return (sentiment + 1) / 2  # Convert to 0-1 scale
            
        except Exception as e:
            self.logger.error(f"Keyword sentiment analysis failed: {e}")
            return 0.5
    
    def _calculate_risk_adjustment(self, text: str) -> float:
        """
        Calculate risk-based adjustment to sentiment score.
        
        Args:
            text: Text to analyze for risk keywords
            
        Returns:
            Risk adjustment factor (negative for high risk)
        """
        try:
            text_lower = text.lower()
            
            risk_count = sum(1 for word in self.risk_keywords if word in text_lower)
            
            if risk_count == 0:
                return 0.0
            
            # Apply negative adjustment for risk keywords
            risk_adjustment = -0.1 * min(risk_count, 3)  # Cap at -0.3
            return risk_adjustment
            
        except Exception as e:
            self.logger.error(f"Risk adjustment calculation failed: {e}")
            return 0.0
    
    def _aggregate_sentiment_scores(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate individual sentiment scores into overall metrics.
        
        Args:
            sentiment_results: List of individual sentiment results
            
        Returns:
            Aggregated sentiment metrics
        """
        try:
            if not sentiment_results:
                return {
                    'overall_sentiment': 0.5,
                    'sentiment_category': 'neutral',
                    'confidence': 0.0
                }
            
            # Calculate weighted average sentiment
            total_weighted_sentiment = 0
            total_weight = 0
            
            for result in sentiment_results:
                weight = max(result.get('engagement', 1), 1)
                sentiment = result.get('weighted_sentiment', 0.5)
                
                total_weighted_sentiment += sentiment * weight
                total_weight += weight
            
            overall_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0.5
            
            # Categorize sentiment
            if overall_sentiment >= 0.7:
                category = 'very_positive'
            elif overall_sentiment >= 0.6:
                category = 'positive'
            elif overall_sentiment >= 0.4:
                category = 'neutral'
            elif overall_sentiment >= 0.3:
                category = 'negative'
            else:
                category = 'very_negative'
            
            # Calculate confidence based on number of mentions and consistency
            confidence = min(len(sentiment_results) / 20, 1.0)  # More mentions = higher confidence
            
            # Reduce confidence if sentiment is highly varied
            sentiments = [r.get('weighted_sentiment', 0.5) for r in sentiment_results]
            sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0
            confidence *= max(0.5, 1 - sentiment_std)  # High variance reduces confidence
            
            return {
                'overall_sentiment': float(overall_sentiment),
                'sentiment_category': category,
                'confidence': float(confidence),
                'mention_breakdown': {
                    'total_mentions': len(sentiment_results),
                    'avg_engagement': float(np.mean([r.get('engagement', 0) for r in sentiment_results])),
                    'sources': list(set(r.get('source', 'unknown') for r in sentiment_results))
                },
                'risk_indicators': {
                    'risk_mentions': sum(1 for r in sentiment_results if r.get('risk_adjustment', 0) < 0),
                    'avg_risk_adjustment': float(np.mean([r.get('risk_adjustment', 0) for r in sentiment_results]))
                }
            }
            
        except Exception as e:
            log_error_with_context(e, {"sentiment_results_count": len(sentiment_results)})
            return {
                'overall_sentiment': 0.5,
                'sentiment_category': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _get_neutral_sentiment_result(self, protocol_name: str, error: str = None) -> Dict[str, Any]:
        """
        Get neutral sentiment result for fallback cases.
        
        Args:
            protocol_name: Protocol name
            error: Optional error message
            
        Returns:
            Neutral sentiment result
        """
        result = {
            'protocol_name': protocol_name,
            'overall_sentiment': 0.5,
            'sentiment_category': 'neutral',
            'confidence': 0.0,
            'mentions_analyzed': 0,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        if error:
            result['error'] = error
        
        return result

    async def get_market_sentiment_overview(self, protocols: List[str]) -> Dict[str, Any]:
        """
        Get overall market sentiment across multiple protocols.
        
        Args:
            protocols: List of protocol names
            
        Returns:
            Market sentiment overview
        """
        try:
            protocol_sentiments = []
            
            for protocol in protocols:
                sentiment = await self.analyze_protocol_sentiment(protocol, days_back=3)
                protocol_sentiments.append(sentiment)
            
            # Calculate market-wide metrics
            overall_sentiments = [s.get('overall_sentiment', 0.5) for s in protocol_sentiments]
            market_sentiment = np.mean(overall_sentiments) if overall_sentiments else 0.5
            
            # Categorize market sentiment
            if market_sentiment >= 0.65:
                market_mood = 'bullish'
            elif market_sentiment >= 0.35:
                market_mood = 'neutral'
            else:
                market_mood = 'bearish'
            
            return {
                'market_sentiment': float(market_sentiment),
                'market_mood': market_mood,
                'protocols_analyzed': len(protocols),
                'individual_sentiments': protocol_sentiments,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            log_error_with_context(e, {"protocols": protocols})
            return {
                'market_sentiment': 0.5,
                'market_mood': 'neutral',
                'error': str(e)
            }
