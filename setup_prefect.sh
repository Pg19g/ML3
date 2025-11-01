#!/bin/bash
# Setup script to configure Prefect for local execution (no server required)

echo "Configuring Prefect for local execution (no server required)..."

# Create .prefect directory if it doesn't exist
mkdir -p ~/.prefect

# Save to .env if it exists
if [ -f .env ]; then
    echo "Updating .env file..."
    
    # Remove old PREFECT settings if they exist
    sed -i.bak '/PREFECT_API_URL/d' .env
    sed -i.bak '/PREFECT_API_DATABASE_CONNECTION_URL/d' .env
    sed -i.bak '/PREFECT_LOGGING_LEVEL/d' .env
    
    # Add new configuration
    echo "" >> .env
    echo "# Prefect Configuration (local mode - no server required)" >> .env
    echo "PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:///~/.prefect/prefect.db" >> .env
    echo "PREFECT_API_URL=" >> .env
    echo "PREFECT_LOGGING_LEVEL=INFO" >> .env
    
    echo "✅ .env file updated"
else
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
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
