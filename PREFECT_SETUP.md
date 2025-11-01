# Prefect Setup Guide

This guide explains how to configure Prefect to run ML3 flows without requiring a Prefect server.

---

## The Issue

By default, Prefect tries to connect to a server at `http://localhost:4200/api/`. If you see this error:

```
RuntimeError: Failed to reach API at http://localhost:4200/api/
```

It means Prefect is trying to connect to a server that isn't running.

---

## Solution: Ephemeral Mode

Prefect can run in **ephemeral mode**, which doesn't require a server. This is perfect for:
- Local development
- Single-user scenarios
- Simple workflows
- Testing

---

## Quick Fix

### Option 1: Automatic Setup (Recommended)

Run the setup script:

```bash
./setup_prefect.sh
```

This will:
1. Configure Prefect to use ephemeral mode
2. Update your `.env` file
3. Set appropriate logging levels

### Option 2: Manual Setup

1. **Update your `.env` file**:

```bash
# Add or update this line in .env
PREFECT_API_URL=ephemeral
```

2. **Or set environment variable**:

```bash
export PREFECT_API_URL="ephemeral"
```

3. **Run your flows**:

```bash
make ingest
make build
make train
```

---

## Verification

To verify Prefect is configured correctly:

```bash
# Check Prefect configuration
python -c "from prefect import get_client; print('Prefect configured correctly!')"
```

If this runs without errors, you're good to go!

---

## Alternative: Run Prefect Server (Optional)

If you want to use the Prefect UI and server features:

### 1. Start Prefect Server

```bash
# In a separate terminal
prefect server start
```

This starts the server at `http://localhost:4200`

### 2. Update .env

```bash
PREFECT_API_URL=http://localhost:4200/api
```

### 3. Access UI

Open browser to: http://localhost:4200

---

## Comparison

| Feature | Ephemeral Mode | Server Mode |
|---------|----------------|-------------|
| Setup | None required | Start server |
| UI | No | Yes (http://localhost:4200) |
| Flow history | No | Yes |
| Scheduling | No | Yes |
| Monitoring | Console only | Full dashboard |
| Multi-user | No | Yes |
| Best for | Local dev, testing | Production, teams |

---

## Recommended Workflow

### Development (Ephemeral)

```bash
# Set ephemeral mode
export PREFECT_API_URL="ephemeral"

# Run flows
make all
```

### Production (Server)

```bash
# Start server (in separate terminal)
prefect server start

# Set server mode
export PREFECT_API_URL="http://localhost:4200/api"

# Run flows
make all
```

---

## Troubleshooting

### Error: "Failed to reach API"

**Solution**: Set ephemeral mode

```bash
export PREFECT_API_URL="ephemeral"
```

Or update `.env`:
```
PREFECT_API_URL=ephemeral
```

### Error: "Connection refused"

**Cause**: Trying to connect to server that isn't running

**Solution**: Either:
1. Use ephemeral mode (recommended for local dev)
2. Start Prefect server: `prefect server start`

### Flows run but no UI

**Cause**: Running in ephemeral mode

**Solution**: This is expected! Ephemeral mode doesn't have a UI. If you want the UI, start the server.

---

## Environment Variables

Add these to your `.env` file:

```bash
# Required
PREFECT_API_URL=ephemeral

# Optional
PREFECT_LOGGING_LEVEL=INFO
PREFECT_LOGGING_TO_API_ENABLED=false
```

---

## For Mac M3 Pro Users

The setup script works on Mac M3 Pro. Just run:

```bash
./setup_prefect.sh
```

Then:

```bash
source .env  # Load environment variables
make all     # Run pipeline
```

---

## Summary

**For most users**: Use ephemeral mode
- ✅ No server required
- ✅ Simple setup
- ✅ Works immediately
- ✅ Perfect for local development

**For advanced users**: Use server mode
- ✅ Full UI
- ✅ Flow history
- ✅ Scheduling
- ✅ Monitoring dashboard

---

*Last Updated: 2024-10-31*
