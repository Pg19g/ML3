#!/bin/bash
# Setup script to configure Prefect for local execution (no server required)

echo "Configuring Prefect for local execution (no server required)..."

# Create .prefect directory if it doesn't exist
mkdir -p ~/.prefect

# Get absolute path to home directory
HOME_DIR=$(eval echo ~)
PREFECT_DB_PATH="${HOME_DIR}/.prefect/prefect.db"

# Save to .env if it exists
if [ -f .env ]; then
    echo "Updating .env file..."
    
    # Remove old PREFECT settings if they exist
    sed -i.bak '/PREFECT_API_URL/d' .env 2>/dev/null || sed -i '' '/PREFECT_API_URL/d' .env
    sed -i.bak '/PREFECT_API_DATABASE_CONNECTION_URL/d' .env 2>/dev/null || sed -i '' '/PREFECT_API_DATABASE_CONNECTION_URL/d' .env
    sed -i.bak '/PREFECT_LOGGING_LEVEL/d' .env 2>/dev/null || sed -i '' '/PREFECT_LOGGING_LEVEL/d' .env
    
    # Add new configuration with absolute path
    echo "" >> .env
    echo "# Prefect Configuration (local mode - no server required)" >> .env
    echo "PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:///${PREFECT_DB_PATH}" >> .env
    echo "PREFECT_API_URL=" >> .env
    echo "PREFECT_LOGGING_LEVEL=INFO" >> .env
    
    echo "✅ .env file updated"
    echo "Database path: ${PREFECT_DB_PATH}"
else
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    
    # Update with absolute path
    sed -i.bak "s|sqlite+aiosqlite:///~/.prefect/prefect.db|sqlite+aiosqlite:///${PREFECT_DB_PATH}|g" .env 2>/dev/null || \
    sed -i '' "s|sqlite+aiosqlite:///~/.prefect/prefect.db|sqlite+aiosqlite:///${PREFECT_DB_PATH}|g" .env
    
    echo "✅ Created .env file. Please add your EODHD_API_KEY"
fi

echo ""
echo "✅ Prefect configured for local execution!"
echo ""
echo "To apply these settings, run:"
echo "  source .env"
echo "  export \$(cat .env | xargs)"
echo ""
echo "Or simply run commands with the settings:"
echo "  make ingest"
echo "  make build"
echo "  make train"
echo "  make all"
