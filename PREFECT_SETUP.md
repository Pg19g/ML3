# Prefect Setup Guide

This guide explains how to configure Prefect to run ML3 flows locally without requiring a Prefect server.

---

## The Issue

By default, Prefect tries to connect to a server. If you see errors like:

```
RuntimeError: Failed to reach API at http://localhost:4200/api/
```

or

```
httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol.
```

It means Prefect is trying to connect to a server that isn't running.

---

## Solution: Local Execution Mode

Prefect can run in **local execution mode** using a local SQLite database, which doesn't require a server. This is perfect for:
- Local development
- Single-user scenarios
- Simple workflows
- Testing

---

## Quick Fix

### Option 1: Automatic Setup (Recommended)

Run the setup script:

```bash
cd /Users/pgalaszek/Desktop/ML3
./setup_prefect.sh
```

Then load the environment variables:

```bash
source .env
export $(cat .env | xargs)
```

### Option 2: Manual Setup

1. **Update your `.env` file**:

Add these lines to `.env`:

```bash
PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:///~/.prefect/prefect.db
PREFECT_API_URL=
PREFECT_LOGGING_LEVEL=INFO
```

2. **Load environment variables**:

```bash
source .env
export $(cat .env | xargs)
```

3. **Run your flows**:

```bash
make ingest
make build
make train
```

---

## Alternative: Set Environment Variables Directly

Instead of using `.env`, you can set environment variables directly:

```bash
export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///~/.prefect/prefect.db"
export PREFECT_API_URL=""
export PREFECT_LOGGING_LEVEL="INFO"

# Then run your commands
make all
```

---

## Makefile Integration

The Makefile commands automatically work with the `.env` file. Just ensure `.env` is configured correctly:

```bash
# Configure once
./setup_prefect.sh

# Then use Makefile commands
make ingest
make build
make train
make backtest
make all
```

---

## Verification

To verify Prefect is configured correctly:

```bash
# Load environment
source .env
export $(cat .env | xargs)

# Test import
python -c "from prefect import flow; print('Prefect configured correctly!')"
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
# Remove or comment out PREFECT_API_DATABASE_CONNECTION_URL
```

### 3. Access UI

Open browser to: http://localhost:4200

---

## Comparison

| Feature | Local Mode | Server Mode |
|---------|------------|-------------|
| Setup | Set env vars | Start server |
| UI | No | Yes (http://localhost:4200) |
| Flow history | Local SQLite | Full database |
| Scheduling | No | Yes |
| Monitoring | Console only | Full dashboard |
| Multi-user | No | Yes |
| Best for | Local dev, testing | Production, teams |

---

## Recommended Workflow

### Development (Local Mode)

```bash
# Configure once
./setup_prefect.sh

# Load environment
source .env
export $(cat .env | xargs)

# Run flows
make all
```

### Production (Server Mode)

```bash
# Start server (in separate terminal)
prefect server start

# Set server mode in .env
PREFECT_API_URL=http://localhost:4200/api

# Run flows
make all
```

---

## Troubleshooting

### Error: "Failed to reach API"

**Solution**: Set local mode

```bash
export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///~/.prefect/prefect.db"
export PREFECT_API_URL=""
```

### Error: "Request URL is missing protocol"

**Cause**: PREFECT_API_URL is set to an invalid value (like "ephemeral")

**Solution**: Set PREFECT_API_URL to empty string

```bash
export PREFECT_API_URL=""
```

### Error: "Connection refused"

**Cause**: Trying to connect to server that isn't running

**Solution**: Either:
1. Use local mode (recommended for local dev)
2. Start Prefect server: `prefect server start`

### Flows run but no UI

**Cause**: Running in local mode

**Solution**: This is expected! Local mode doesn't have a UI. If you want the UI, start the server.

---

## Environment Variables

Add these to your `.env` file for local mode:

```bash
# Required for local mode
PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:///~/.prefect/prefect.db
PREFECT_API_URL=

# Optional
PREFECT_LOGGING_LEVEL=INFO
```

---

## For Mac M3 Pro Users

The setup script works on Mac M3 Pro. Just run:

```bash
cd /Users/pgalaszek/Desktop/ML3
./setup_prefect.sh

# Load environment
source .env
export $(cat .env | xargs)

# Run pipeline
make all
```

---

## Summary

**For most users**: Use local mode
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
