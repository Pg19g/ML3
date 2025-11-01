# Quick Fix for Prefect Connection Error

## The Problem

You saw this error:
```
RuntimeError: Failed to reach API at http://localhost:4200/api/
```

## The Solution (2 minutes)

### Option 1: Automatic (Recommended)

```bash
cd /Users/pgalaszek/Desktop/ML3
./setup_prefect.sh
```

Done! Now run:
```bash
make all
```

### Option 2: Manual

Edit your `.env` file:

```bash
cd /Users/pgalaszek/Desktop/ML3
nano .env  # or use your preferred editor
```

Add or update this line:
```
PREFECT_API_URL=ephemeral
```

Save and exit. Then run:
```bash
make all
```

## Verification

To verify it's working:

```bash
# This should run without errors
python -c "import os; os.environ['PREFECT_API_URL']='ephemeral'; from prefect import flow; print('Success!')"
```

## What This Does

- Configures Prefect to run in "ephemeral mode"
- No Prefect server needed
- Flows run locally
- Perfect for development and testing

## Full Documentation

See [PREFECT_SETUP.md](PREFECT_SETUP.md) for complete details.

---

**TL;DR**: Run `./setup_prefect.sh` and you're good to go!
