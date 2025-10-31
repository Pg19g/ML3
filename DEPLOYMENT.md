# ML3 Deployment Guide

This guide covers deploying ML3 in various environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring](#monitoring)
6. [Maintenance](#maintenance)

## Local Development

### Quick Setup

```bash
# Clone repository
git clone https://github.com/Pg19g/ML3.git
cd ML3

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your EODHD_API_KEY

# Run pipeline
make ingest
make build
make train

# Launch dashboard
make dashboard
```

### Development Workflow

1. **Make changes** to source code
2. **Run tests**: `make test`
3. **Format code**: `make format`
4. **Check linting**: `make lint`
5. **Commit changes**: `git commit -am "Description"`
6. **Push to GitHub**: `git push`

## Production Deployment

### System Requirements

- **OS**: Ubuntu 22.04 LTS or similar
- **Python**: 3.11+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ for data
- **CPU**: 4+ cores recommended

### Installation Steps

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip -y

# Clone repository
git clone https://github.com/Pg19g/ML3.git
cd ML3

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add EODHD_API_KEY

# Create data directories
mkdir -p data/{raw,pit,samples}
mkdir -p models/registry
mkdir -p reports

# Set permissions
chmod +x src/cli.py
```

### Running as a Service

Create systemd service files:

**Dashboard Service** (`/etc/systemd/system/ml3-dashboard.service`):

```ini
[Unit]
Description=ML3 Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ML3
Environment="PATH=/home/ubuntu/ML3/venv/bin"
ExecStart=/home/ubuntu/ML3/venv/bin/python -m src.cli dashboard --port 8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**API Service** (`/etc/systemd/system/ml3-api.service`):

```ini
[Unit]
Description=ML3 FastAPI Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ML3
Environment="PATH=/home/ubuntu/ML3/venv/bin"
ExecStart=/home/ubuntu/ML3/venv/bin/python -m src.cli api --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ml3-dashboard ml3-api
sudo systemctl start ml3-dashboard ml3-api

# Check status
sudo systemctl status ml3-dashboard
sudo systemctl status ml3-api
```

### Nginx Reverse Proxy

Install Nginx:

```bash
sudo apt install nginx -y
```

Configure Nginx (`/etc/nginx/sites-available/ml3`):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Dashboard
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/ml3 /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

### Scheduled Data Updates

Add cron jobs for automated data refresh:

```bash
crontab -e
```

Add these lines:

```cron
# Refresh prices daily at 6 AM
0 6 * * * cd /home/ubuntu/ML3 && /home/ubuntu/ML3/venv/bin/python -m src.cli data ingest-prices --incremental

# Refresh fundamentals weekly on Sunday at 7 AM
0 7 * * 0 cd /home/ubuntu/ML3 && /home/ubuntu/ML3/venv/bin/python -m src.cli data ingest-fundamentals --incremental

# Rebuild features daily at 8 AM
0 8 * * * cd /home/ubuntu/ML3 && /home/ubuntu/ML3/venv/bin/python -m src.cli data build-pit && /home/ubuntu/ML3/venv/bin/python -m src.cli features build && /home/ubuntu/ML3/venv/bin/python -m src.cli labels build
```

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/{raw,pit,samples} models/registry reports

# Expose ports
EXPOSE 8501 8000

# Default command
CMD ["python", "-m", "src.cli", "dashboard"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - EODHD_API_KEY=${EODHD_API_KEY}
      - TZ=UTC
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
    command: python -m src.cli dashboard --port 8501
    restart: unless-stopped

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - EODHD_API_KEY=${EODHD_API_KEY}
      - TZ=UTC
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
    command: python -m src.cli api --host 0.0.0.0 --port 8000
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Cloud Deployment

### AWS EC2

1. **Launch EC2 Instance**:
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t3.medium or larger
   - Storage: 50GB+ EBS volume
   - Security Group: Allow ports 80, 443, 8000, 8501

2. **Connect and Setup**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Follow production deployment steps above**

4. **Configure Elastic IP** for static IP address

5. **Set up CloudWatch** for monitoring

### Google Cloud Platform

1. **Create Compute Engine Instance**:
   - Machine type: n1-standard-2 or larger
   - Boot disk: Ubuntu 22.04 LTS, 50GB
   - Firewall: Allow HTTP, HTTPS

2. **Connect via SSH**

3. **Follow production deployment steps**

4. **Set up Cloud Monitoring**

### Azure

1. **Create Virtual Machine**:
   - Image: Ubuntu 22.04 LTS
   - Size: Standard_B2s or larger
   - Disk: 50GB

2. **Configure Network Security Group**:
   - Allow ports 80, 443, 8000, 8501

3. **Follow production deployment steps**

4. **Set up Azure Monitor**

## Monitoring

### Application Logs

View service logs:

```bash
# Dashboard logs
sudo journalctl -u ml3-dashboard -f

# API logs
sudo journalctl -u ml3-api -f
```

### System Monitoring

Install monitoring tools:

```bash
# Install htop for system monitoring
sudo apt install htop -y

# Install disk usage analyzer
sudo apt install ncdu -y
```

### Performance Metrics

Monitor key metrics:

- **CPU Usage**: Should be < 80% average
- **Memory Usage**: Should be < 80% of available RAM
- **Disk Usage**: Keep < 80% full
- **API Response Time**: Should be < 1 second
- **Data Freshness**: Check last update timestamps

### Health Checks

Create health check script (`scripts/health_check.sh`):

```bash
#!/bin/bash

# Check dashboard
if curl -s http://localhost:8501 > /dev/null; then
    echo "✓ Dashboard is running"
else
    echo "✗ Dashboard is down"
fi

# Check API
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ API is running"
else
    echo "✗ API is down"
fi

# Check data freshness
LAST_UPDATE=$(stat -c %Y data/raw/prices_daily.parquet 2>/dev/null || echo 0)
NOW=$(date +%s)
AGE=$((NOW - LAST_UPDATE))
DAYS=$((AGE / 86400))

if [ $DAYS -lt 7 ]; then
    echo "✓ Data is fresh ($DAYS days old)"
else
    echo "⚠ Data is stale ($DAYS days old)"
fi
```

Run health checks:

```bash
chmod +x scripts/health_check.sh
./scripts/health_check.sh
```

### Alerting

Set up email alerts with `mailutils`:

```bash
sudo apt install mailutils -y

# Create alert script
cat > scripts/alert.sh << 'EOF'
#!/bin/bash
if ! systemctl is-active --quiet ml3-dashboard; then
    echo "ML3 Dashboard is down!" | mail -s "ML3 Alert" your-email@example.com
fi
EOF

chmod +x scripts/alert.sh

# Add to crontab
crontab -e
# Add: */5 * * * * /home/ubuntu/ML3/scripts/alert.sh
```

## Maintenance

### Backup Strategy

**Daily Backups**:

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backup/ml3"
DATE=$(date +%Y%m%d)

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Backup config
tar -czf $BACKUP_DIR/config_$DATE.tar.gz config/

# Keep only last 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

**Automated Backups**:

```cron
# Daily backup at 2 AM
0 2 * * * /home/ubuntu/ML3/scripts/backup.sh
```

### Update Procedure

```bash
# 1. Backup current version
./scripts/backup.sh

# 2. Pull latest changes
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Restart services
sudo systemctl restart ml3-dashboard ml3-api

# 5. Verify
./scripts/health_check.sh
```

### Database Maintenance

Clean up old data:

```bash
# Remove old reports (keep last 90 days)
find reports/ -name "*.json" -mtime +90 -delete
find reports/ -name "*.parquet" -mtime +90 -delete

# Optimize parquet files
python -c "
import pandas as pd
from pathlib import Path

for file in Path('data/pit').glob('*.parquet'):
    df = pd.read_parquet(file)
    df.to_parquet(file, compression='snappy', index=False)
"
```

### Security Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
pip list --outdated
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Check for vulnerabilities
pip install safety
safety check
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u ml3-dashboard -n 50
sudo journalctl -u ml3-api -n 50

# Check permissions
ls -la /home/ubuntu/ML3

# Verify Python environment
source venv/bin/activate
python --version
pip list
```

### High Memory Usage

```bash
# Check memory
free -h

# Identify process
top -o %MEM

# Restart services
sudo systemctl restart ml3-dashboard ml3-api
```

### Slow Performance

```bash
# Check disk I/O
iostat -x 1

# Check network
netstat -tuln

# Optimize database
# See Database Maintenance section
```

## Best Practices

1. **Regular Backups**: Daily automated backups
2. **Monitoring**: Set up alerts for failures
3. **Updates**: Keep system and packages updated
4. **Security**: Use SSL, firewall, and strong passwords
5. **Documentation**: Document all changes
6. **Testing**: Test updates in staging first
7. **Logs**: Rotate logs to prevent disk full
8. **Capacity Planning**: Monitor growth trends

## Support

For deployment issues:
- Check logs first
- Review this guide
- GitHub Issues: [repository-url]/issues
- Email: support@example.com

---

**Security Note**: Never commit `.env` file or API keys to version control. Use environment variables or secrets management.
