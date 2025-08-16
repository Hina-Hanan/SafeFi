"""
Unit tests for validation utilities.

This module tests all validation functions including address validation,
API response validation, and protocol data validation.
"""

import pytest
from unittest.mock import patch

from src.utils.validators import (
    validate_ethereum_address,
    validate_api_response,
    validate_protocol_data,
    validate_risk_score,
    ValidationError
)


class TestValidators:
    """Test cases for validation functions."""
    
    def test_validate_ethereum_address_valid(self):
        """Test valid Ethereum address validation."""
        valid_address = "0x742d35Cc6634C0532925a3b8d93f97a2D71D2c8e"
        
        with patch('src.utils.validators.Web3.isAddress', return_value=True):
            result = validate_ethereum_address(valid_address)
            assert result is True
    
    def test_validate_ethereum_address_invalid_format(self):
        """Test invalid Ethereum address format."""
        invalid_address = "invalid_address"
        
        with pytest.raises(ValidationError, match="Address must start with '0x'"):
            validate_ethereum_address(invalid_address)
    
    def test_validate_ethereum_address_wrong_length(self):
        """Test Ethereum address with wrong length."""
        short_address = "0x742d35Cc"
        
        with pytest.raises(ValidationError, match="Address must be 42 characters long"):
            validate_ethereum_address(short_address)
    
    def test_validate_api_response_valid(self):
        """Test valid API response validation."""
        response = {
            "name": "test",
            "tvl": 1000000,
            "category": "lending"
        }
        required_fields = ["name", "tvl", "category"]
        
        result = validate_api_response(response, required_fields)
        assert result is True
    
    def test_validate_api_response_missing_fields(self):
        """Test API response with missing fields."""
        response = {"name": "test"}
        required_fields = ["name", "tvl", "category"]
        
        with pytest.raises(ValidationError, match="Missing required fields"):
            validate_api_response(response, required_fields)
    
    def test_validate_protocol_data_valid(self):
        """Test valid protocol data validation."""
        protocol_data = {
            "name": "Uniswap",
            "tvl": 5000000000,
            "category": "dex",
            "chain": "Ethereum"
        }
        
        result = validate_protocol_data(protocol_data)
        assert result is True
    
    def test_validate_protocol_data_negative_tvl(self):
        """Test protocol data with negative TVL."""
        protocol_data = {
            "name": "Test Protocol",
            "tvl": -100,
            "category": "lending",
            "chain": "Ethereum"
        }
        
        with pytest.raises(ValidationError, match="TVL must be a non-negative number"):
            validate_protocol_data(protocol_data)
    
    def test_validate_risk_score_valid(self):
        """Test valid risk score validation."""
        assert validate_risk_score(50.5) is True
        assert validate_risk_score(0) is True
        assert validate_risk_score(100) is True
    
    def test_validate_risk_score_out_of_range(self):
        """Test risk score out of valid range."""
        with pytest.raises(ValidationError, match="Risk score must be between 0 and 100"):
            validate_risk_score(-1)
        
        with pytest.raises(ValidationError, match="Risk score must be between 0 and 100"):
            validate_risk_score(101)
    
    def test_validate_risk_score_invalid_type(self):
        """Test risk score with invalid type."""
        with pytest.raises(ValidationError, match="Risk score must be a number"):
            validate_risk_score("invalid")
