# Quick Fix for Prefect Database Path Error

## The Problem

You saw this error:
```
sqlite3.OperationalError: unable to open database file
```

This happens because the tilde (`~`) in the database path isn't being expanded.

## The Solution (1 minute)

### Step 1: Run Setup Script

```bash
cd /Users/pgalaszek/Desktop/ML3
./setup_prefect.sh
```

This will automatically create the correct path with your home directory.

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

### 2. Add these lines (replace /Users/pgalaszek with your actual home directory)

```
PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:////Users/pgalaszek/.prefect/prefect.db
PREFECT_API_URL=
PREFECT_LOGGING_LEVEL=INFO
```

**Important**: 
- Use **4 slashes** after `sqlite+aiosqlite:` (one for protocol, three for absolute path)
- Use your **actual home directory path** (not `~`)
- `PREFECT_API_URL` should be empty (nothing after the `=`)

### 3. Create the directory

```bash
mkdir -p ~/.prefect
```

### 4. Save and load

```bash
source .env
export $(cat .env | xargs)
```

### 5. Run

```bash
make all
```

---

## One-Liner Fix

```bash
cd /Users/pgalaszek/Desktop/ML3 && mkdir -p ~/.prefect && export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:////Users/pgalaszek/.prefect/prefect.db" && export PREFECT_API_URL="" && make all
```

---

## What This Does

- Creates `.prefect` directory in your home folder
- Configures Prefect to use a local SQLite database with absolute path
- Disables API server connection
- Flows run locally without a server

---

## Verification

Test if it's working:

```bash
# Check if directory exists
ls -la ~/.prefect

# Load environment
source .env
export $(cat .env | xargs)

# Test
python -c "from prefect import flow; print('Success!')"
```

If no errors, you're good!

---

## Why This Happened

SQLite doesn't expand the tilde (`~`) character to your home directory. You need to use either:
1. The `$HOME` environment variable
2. The absolute path (e.g., `/Users/pgalaszek/`)

The setup script automatically detects your home directory and creates the correct path.

---

## Full Documentation

See [PREFECT_SETUP.md](PREFECT_SETUP.md) for complete details.

---

**TL;DR**: 
1. Run `./setup_prefect.sh`
2. Run `source .env && export $(cat .env | xargs)`
3. Run `make all`
