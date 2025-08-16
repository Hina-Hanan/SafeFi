"""
Validation utilities for SafeFi DeFi Risk Assessment Agent.

This module provides validation functions for various data types,
API responses, and blockchain-related data.
"""

import re
from typing import Any, Dict, Optional, List
from web3 import Web3
from loguru import logger


class ValidationError(Exception):
    """Custom validation error exception."""
    pass


def validate_ethereum_address(address: str) -> bool:
    """
    Validate Ethereum address format.
    
    Args:
        address: Ethereum address to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValidationError: If address is invalid
    """
    try:
        if not isinstance(address, str):
            raise ValidationError("Address must be a string")
        
        if not address.startswith('0x'):
            raise ValidationError("Address must start with '0x'")
        
        if len(address) != 42:
            raise ValidationError("Address must be 42 characters long")
        
        # Check if it's a valid hex string
        int(address, 16)
        
        # Use Web3 for checksum validation
        if Web3.isAddress(address):
            logger.debug(f"Valid Ethereum address: {address}")
            return True
        else:
            raise ValidationError("Invalid Ethereum address checksum")
            
    except ValueError as e:
        raise ValidationError(f"Invalid hex format in address: {str(e)}")
    except Exception as e:
        logger.error(f"Address validation error: {str(e)}")
        raise ValidationError(f"Address validation failed: {str(e)}")


def validate_api_response(response: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate API response structure.
    
    Args:
        response: API response dictionary
        required_fields: List of required fields
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValidationError: If response is invalid
    """
    try:
        if not isinstance(response, dict):
            raise ValidationError("Response must be a dictionary")
        
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")
        
        logger.debug("API response validation successful")
        return True
        
    except Exception as e:
        logger.error(f"API response validation error: {str(e)}")
        raise ValidationError(f"API response validation failed: {str(e)}")


def validate_protocol_data(protocol_data: Dict[str, Any]) -> bool:
    """
    Validate DeFi protocol data structure.
    
    Args:
        protocol_data: Protocol data dictionary
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValidationError: If protocol data is invalid
    """
    required_fields = ["name", "tvl", "category", "chain"]
    
    try:
        validate_api_response(protocol_data, required_fields)
        
        # Additional validations
        if not isinstance(protocol_data["tvl"], (int, float)) or protocol_data["tvl"] < 0:
            raise ValidationError("TVL must be a non-negative number")
        
        if not isinstance(protocol_data["name"], str) or len(protocol_data["name"]) == 0:
            raise ValidationError("Protocol name must be a non-empty string")
        
        logger.debug(f"Protocol data validation successful for: {protocol_data['name']}")
        return True
        
    except Exception as e:
        logger.error(f"Protocol data validation error: {str(e)}")
        raise ValidationError(f"Protocol data validation failed: {str(e)}")


def validate_risk_score(risk_score: float) -> bool:
    """
    Validate risk score value.
    
    Args:
        risk_score: Risk score to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValidationError: If risk score is invalid
    """
    try:
        if not isinstance(risk_score, (int, float)):
            raise ValidationError("Risk score must be a number")
        
        if not 0 <= risk_score <= 100:
            raise ValidationError("Risk score must be between 0 and 100")
        
        logger.debug(f"Risk score validation successful: {risk_score}")
        return True
        
    except Exception as e:
        logger.error(f"Risk score validation error: {str(e)}")
        raise ValidationError(f"Risk score validation failed: {str(e)}")
