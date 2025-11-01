# Quick Fix for Prefect Connection Error

## The Problem

You saw this error:
```
RuntimeError: Failed to reach API at ephemeral/
```

or

```
httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol.
```

## The Solution (2 minutes)

### Step 1: Run Setup Script

```bash
cd /Users/pgalaszek/Desktop/ML3
./setup_prefect.sh
```

### Step 2: Load Environment Variables

```bash
source .env
export $(cat .env | xargs)
```

### Step 3: Run Pipeline

```bash
make all
```

Done!

---

## Manual Alternative

If the script doesn't work, do this manually:

### 1. Edit .env file

```bash
cd /Users/pgalaszek/Desktop/ML3
nano .env  # or use your preferred editor
```

### 2. Add these lines

```
PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:///~/.prefect/prefect.db
PREFECT_API_URL=
PREFECT_LOGGING_LEVEL=INFO
```

**Important**: `PREFECT_API_URL` should be set to empty (nothing after the `=`)

### 3. Save and load

```bash
source .env
export $(cat .env | xargs)
```

### 4. Run

```bash
make all
```

---

## One-Liner Fix

```bash
cd /Users/pgalaszek/Desktop/ML3 && export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///~/.prefect/prefect.db" && export PREFECT_API_URL="" && make all
```

---

## What This Does

- Configures Prefect to use a local SQLite database
- Disables API server connection (sets `PREFECT_API_URL` to empty)
- Flows run locally without a server
- Perfect for development and testing

---

## Verification

Test if it's working:

```bash
# Load environment
source .env
export $(cat .env | xargs)

# Test
python -c "from prefect import flow; print('Success!')"
```

If no errors, you're good!

---

## Full Documentation

See [PREFECT_SETUP.md](PREFECT_SETUP.md) for complete details.

---

**TL;DR**: 
1. Run `./setup_prefect.sh`
2. Run `source .env && export $(cat .env | xargs)`
3. Run `make all`
