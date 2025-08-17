import asyncio
import asyncpg
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def test_postgres_connection():
    """Test PostgreSQL connection using SafeFi configuration."""
    
    print("🔍 Testing PostgreSQL connection...")
    print("-" * 50)
    
    try:
        # Use environment variables or defaults for port 5433
        host = os.getenv('DB_HOST', 'localhost')
        port = int(os.getenv('DB_PORT', 5433))  # Default to 5433
        database = os.getenv('DB_NAME', 'safefi_db')
        user = os.getenv('DB_USER', 'safefi_user')
        password = os.getenv('DB_PASSWORD', 'your_secure_password')
        
        print(f"Connecting to: {user}@{host}:{port}/{database}")
        
        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        # Test basic query
        version = await conn.fetchval('SELECT version()')
        
        # Test if user can create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                test_data TEXT
            )
        """)
        
        # Insert test data
        await conn.execute(
            "INSERT INTO test_table (test_data) VALUES ($1)",
            "SafeFi PostgreSQL Test"
        )
        
        # Retrieve test data
        result = await conn.fetchval(
            "SELECT test_data FROM test_table ORDER BY id DESC LIMIT 1"
        )
        
        # Clean up test table
        await conn.execute("DROP TABLE test_table")
        
        # Close connection
        await conn.close()
        
        print("✅ PostgreSQL connection successful!")
        print(f"✅ PostgreSQL version: {version[:50]}...")
        print(f"✅ Database operations working: {result}")
        print(f"✅ SafeFi can connect to PostgreSQL on port {port}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("\n🔧 Check the following:")
        print("  1. PostgreSQL is running")
        print("  2. Port 5433 is correct")
        print("  3. Database 'safefi_db' exists")
        print("  4. User 'safefi_user' exists with correct password")
        print("  5. .env file has correct DATABASE_URL")
        
        return False

if __name__ == '__main__':
    asyncio.run(test_postgres_connection())
