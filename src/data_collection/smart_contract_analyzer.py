"""
Smart Contract Analysis module for SafeFi DeFi Risk Assessment Agent.

This module provides smart contract risk assessment capabilities
including audit status, code complexity, and deployment age analysis.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from loguru import logger

from ..utils.logger import get_logger, log_error_with_context
from ..utils.validators import validate_ethereum_address, ValidationError


class SmartContractAnalyzer:
    """
    Smart contract analysis and risk assessment.
    
    This class provides methods to analyze smart contract risk factors
    including deployment age, audit status, and complexity metrics.
    """
    
    def __init__(self):
        """Initialize smart contract analyzer."""
        self.logger = get_logger("SmartContractAnalyzer")
        
        # Protocol risk profiles based on known audits and history
        self.protocol_risk_profiles = {
            'uniswap': {
                'audit_score': 9.5,
                'deployment_age_months': 60,
                'complexity_score': 7,
                'governance_maturity': 9,
                'exploit_history': 0
            },
            'aave': {
                'audit_score': 9.0,
                'deployment_age_months': 48,
                'complexity_score': 8,
                'governance_maturity': 8,
                'exploit_history': 1
            },
            'compound': {
                'audit_score': 9.0,
                'deployment_age_months': 54,
                'complexity_score': 6,
                'governance_maturity': 8,
                'exploit_history': 0
            },
            'curve': {
                'audit_score': 8.0,
                'deployment_age_months': 42,
                'complexity_score': 8,
                'governance_maturity': 7,
                'exploit_history': 1
            },
            'balancer': {
                'audit_score': 8.5,
                'deployment_age_months': 38,
                'complexity_score': 9,
                'governance_maturity': 7,
                'exploit_history': 2
            }
        }
    
    async def analyze_contract_risk(self, protocol_name: str, 
                                  contract_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze smart contract risk for a protocol.
        
        Args:
            protocol_name: Name of the protocol
            contract_address: Optional contract address for validation
            
        Returns:
            Dictionary containing smart contract risk analysis
        """
        try:
            # Validate contract address if provided
            if contract_address:
                validate_ethereum_address(contract_address)
            
            # Get protocol risk profile
            profile = self.protocol_risk_profiles.get(
                protocol_name.lower(), 
                self._get_default_risk_profile()
            )
            
            # Calculate composite risk scores
            risk_analysis = await self._calculate_risk_scores(protocol_name, profile)
            
            # Add metadata
            risk_analysis.update({
                'protocol_name': protocol_name,
                'contract_address': contract_address,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'analyzer_version': '1.0.0'
            })
            
            self.logger.info(f"Completed smart contract analysis for {protocol_name}")
            return risk_analysis
            
        except ValidationError as e:
            self.logger.error(f"Validation error in contract analysis: {e}")
            raise
        except Exception as e:
            log_error_with_context(e, {
                "protocol_name": protocol_name,
                "contract_address": contract_address
            })
            raise
    
    def _get_default_risk_profile(self) -> Dict[str, float]:
        """
        Get default risk profile for unknown protocols.
        
        Returns:
            Default risk profile dictionary
        """
        return {
            'audit_score': 5.0,  # Neutral/unknown audit status
            'deployment_age_months': 6,  # Assume relatively new
            'complexity_score': 7,  # Medium complexity
            'governance_maturity': 5,  # Unknown governance
            'exploit_history': 0  # No known exploits
        }
    
    async def _calculate_risk_scores(self, protocol_name: str, 
                                   profile: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate comprehensive risk scores from profile data.
        
        Args:
            protocol_name: Protocol name
            profile: Risk profile data
            
        Returns:
            Dictionary containing calculated risk scores
        """
        try:
            scores = {}
            
            # Technical Risk (based on audit score and complexity)
            technical_risk = self._calculate_technical_risk(
                profile['audit_score'],
                profile['complexity_score']
            )
            scores['technical_risk'] = technical_risk
            
            # Maturity Risk (based on deployment age)
            maturity_risk = self._calculate_maturity_risk(
                profile['deployment_age_months']
            )
            scores['maturity_risk'] = maturity_risk
            
            # Governance Risk
            governance_risk = self._calculate_governance_risk(
                profile['governance_maturity']
            )
            scores['governance_risk'] = governance_risk
            
            # Historical Risk (based on exploit history)
            historical_risk = self._calculate_historical_risk(
                profile['exploit_history']
            )
            scores['historical_risk'] = historical_risk
            
            # Overall Smart Contract Risk (weighted average)
            overall_risk = (
                technical_risk * 0.35 +
                maturity_risk * 0.20 +
                governance_risk * 0.25 +
                historical_risk * 0.20
            )
            scores['overall_smart_contract_risk'] = overall_risk
            
            # Risk level categorization
            scores['risk_level'] = self._categorize_risk_level(overall_risk)
            
            # Individual component scores for explainability
            scores['components'] = {
                'audit_score': profile['audit_score'],
                'deployment_age_months': profile['deployment_age_months'],
                'complexity_score': profile['complexity_score'],
                'governance_maturity': profile['governance_maturity'],
                'exploit_history': profile['exploit_history']
            }
            
            # Confidence score based on data availability
            scores['confidence'] = self._calculate_confidence_score(protocol_name)
            
            return scores
            
        except Exception as e:
            log_error_with_context(e, {"protocol_name": protocol_name, "profile": profile})
            raise
    
    def _calculate_technical_risk(self, audit_score: float, complexity_score: float) -> float:
        """Calculate technical risk score."""
        # Higher audit score = lower risk, higher complexity = higher risk
        audit_risk = (10 - audit_score) / 10  # Invert audit score
        complexity_risk = complexity_score / 10
        
        # Weighted combination
        technical_risk = (audit_risk * 0.7 + complexity_risk * 0.3)
        return min(max(technical_risk, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_maturity_risk(self, deployment_age_months: float) -> float:
        """Calculate maturity risk score based on deployment age."""
        # Newer protocols have higher risk
        if deployment_age_months >= 36:
            return 0.1  # Very mature
        elif deployment_age_months >= 24:
            return 0.3  # Mature
        elif deployment_age_months >= 12:
            return 0.5  # Medium maturity
        elif deployment_age_months >= 6:
            return 0.7  # Young
        else:
            return 0.9  # Very new/high risk
    
    def _calculate_governance_risk(self, governance_maturity: float) -> float:
        """Calculate governance risk score."""
        # Higher governance maturity = lower risk
        governance_risk = (10 - governance_maturity) / 10
        return min(max(governance_risk, 0.0), 1.0)
    
    def _calculate_historical_risk(self, exploit_history: int) -> float:
        """Calculate historical risk based on past exploits."""
        # More exploits = higher risk
        if exploit_history == 0:
            return 0.1  # No known exploits
        elif exploit_history == 1:
            return 0.4  # One exploit
        elif exploit_history == 2:
            return 0.6  # Two exploits
        else:
            return 0.8  # Multiple exploits
    
    def _categorize_risk_level(self, overall_risk: float) -> str:
        """Categorize overall risk into human-readable levels."""
        if overall_risk <= 0.25:
            return "Low"
        elif overall_risk <= 0.5:
            return "Medium"
        elif overall_risk <= 0.75:
            return "High"
        else:
            return "Critical"
    
    def _calculate_confidence_score(self, protocol_name: str) -> float:
        """Calculate confidence score based on data availability."""
        # Higher confidence for well-known protocols
        if protocol_name.lower() in self.protocol_risk_profiles:
            return 0.9  # High confidence - we have specific data
        else:
            return 0.5  # Medium confidence - using defaults
    
    async def get_protocol_audit_status(self, protocol_name: str) -> Dict[str, Any]:
        """
        Get detailed audit status for a protocol.
        
        Args:
            protocol_name: Protocol name
            
        Returns:
            Dictionary containing audit information
        """
        try:
            profile = self.protocol_risk_profiles.get(protocol_name.lower())
            
            if profile:
                audit_info = {
                    'has_audit_data': True,
                    'audit_score': profile['audit_score'],
                    'audit_quality': self._get_audit_quality_description(profile['audit_score']),
                    'last_audit_estimated': "Recent" if profile['audit_score'] > 8 else "Unknown",
                    'auditor_reputation': "High" if profile['audit_score'] > 8.5 else "Medium"
                }
            else:
                audit_info = {
                    'has_audit_data': False,
                    'audit_score': 5.0,
                    'audit_quality': "Unknown",
                    'last_audit_estimated': "Unknown",
                    'auditor_reputation': "Unknown"
                }
            
            audit_info['protocol_name'] = protocol_name
            return audit_info
            
        except Exception as e:
            log_error_with_context(e, {"protocol_name": protocol_name})
            return {'error': str(e)}
    
    def _get_audit_quality_description(self, audit_score: float) -> str:
        """Get human-readable audit quality description."""
        if audit_score >= 9:
            return "Excellent"
        elif audit_score >= 8:
            return "Good"
        elif audit_score >= 7:
            return "Fair"
        elif audit_score >= 6:
            return "Poor"
        else:
            return "Very Poor"
