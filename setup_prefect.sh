#!/bin/bash
# Setup script to configure Prefect for ephemeral mode

echo "Configuring Prefect for ephemeral mode (no server required)..."

# Set Prefect to use ephemeral API
export PREFECT_API_URL="ephemeral"

# Optionally disable telemetry
export PREFECT_LOGGING_LEVEL="INFO"

# Save to .env if it exists
if [ -f .env ]; then
    echo "Updating .env file..."
    
    # Remove old PREFECT_API_URL if exists
    sed -i.bak '/PREFECT_API_URL/d' .env
    
    # Add new configuration
    echo "" >> .env
    echo "# Prefect Configuration (ephemeral mode - no server required)" >> .env
    echo "PREFECT_API_URL=ephemeral" >> .env
    echo "PREFECT_LOGGING_LEVEL=INFO" >> .env
    
    echo "✅ .env file updated"
else
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file. Please add your EODHD_API_KEY"
fi

echo ""
echo "✅ Prefect configured for ephemeral mode!"
echo ""
echo "You can now run:"
echo "  make ingest"
echo "  make build"
echo "  make train"
echo "  make all"
