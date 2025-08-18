# api_key_test.py
import asyncio
import aiohttp
from typing import Dict, Any
import json

async def test_coingecko_api(api_key: str = None) -> Dict[str, Any]:
    """Test CoinGecko API key"""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ethereum",
        "vs_currencies": "usd"
    }
    
    headers = {}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "✅ Working",
                        "data": data,
                        "rate_limit": response.headers.get("X-RateLimit-Remaining", "Unknown")
                    }
                else:
                    return {
                        "status": f"❌ Failed ({response.status})",
                        "error": await response.text()
                    }
    except Exception as e:
        return {"status": f"❌ Error: {str(e)}"}

async def test_etherscan_api(api_key: str) -> Dict[str, Any]:
    """Test Etherscan API key"""
    url = "https://api.etherscan.io/api"
    params = {
        "module": "stats",
        "action": "ethprice",
        "apikey": api_key
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "1":
                        return {
                            "status": "✅ Working",
                            "eth_price": data["result"]["ethusd"],
                            "data": data
                        }
                    else:
                        return {
                            "status": "❌ API Error",
                            "message": data.get("message", "Unknown error")
                        }
                else:
                    return {
                        "status": f"❌ HTTP Error ({response.status})",
                        "error": await response.text()
                    }
    except Exception as e:
        return {"status": f"❌ Error: {str(e)}"}

async def test_defillama_api() -> Dict[str, Any]:
    """Test DeFiLlama API (no key required)"""
    url = "https://api.llama.fi/protocols"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "✅ Working",
                        "protocols_count": len(data),
                        "sample": data[0]["name"] if data else "No data"
                    }
                else:
                    return {
                        "status": f"❌ Failed ({response.status})",
                        "error": await response.text()
                    }
    except Exception as e:
        return {"status": f"❌ Error: {str(e)}"}

async def test_all_apis():
    """Test all API keys"""
    print("🔍 TESTING API KEYS...")
    print("=" * 50)
    
    # Load your settings (modify path as needed)
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        
        # Test CoinGecko
        print("\n📊 Testing CoinGecko API:")
        coingecko_result = await test_coingecko_api(settings.coingecko_api_key)
        print(f"Status: {coingecko_result['status']}")
        if "data" in coingecko_result:
            print(f"BTC Price: ${coingecko_result['data'].get('bitcoin', {}).get('usd', 'N/A')}")
        
        # Test Etherscan
        print("\n⛽ Testing Etherscan API:")
        if settings.etherscan_api_key:
            etherscan_result = await test_etherscan_api(settings.etherscan_api_key)
            print(f"Status: {etherscan_result['status']}")
            if "eth_price" in etherscan_result:
                print(f"ETH Price: ${etherscan_result['eth_price']}")
        else:
            print("⚠️  No API key configured")
        
        # Test DeFiLlama
        print("\n🦙 Testing DeFiLlama API:")
        defillama_result = await test_defillama_api()
        print(f"Status: {defillama_result['status']}")
        if "protocols_count" in defillama_result:
            print(f"Protocols available: {defillama_result['protocols_count']}")
            
    except ImportError:
        print("❌ Could not import settings. Testing with manual configuration...")
        
        # Manual testing if settings import fails
        coingecko_result = await test_coingecko_api()
        print(f"CoinGecko (no key): {coingecko_result['status']}")
        
        defillama_result = await test_defillama_api()
        print(f"DeFiLlama: {defillama_result['status']}")

if __name__ == "__main__":
    asyncio.run(test_all_apis())
