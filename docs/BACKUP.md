# AutoCognitix Backup and Data Management Guide

This document describes the comprehensive backup, export, import, and data synchronization tools available in AutoCognitix.

## Table of Contents

- [Overview](#overview)
- [Backup Script](#backup-script)
- [Export Script](#export-script)
- [Import Script](#import-script)
- [Data Sync Script](#data-sync-script)
- [Cloud Storage Integration](#cloud-storage-integration)
- [Scheduled Backups (Cron)](#scheduled-backups-cron)
- [Disaster Recovery](#disaster-recovery)
- [Best Practices](#best-practices)

## Overview

AutoCognitix uses three primary data stores:

| Database | Purpose | Data Type |
|----------|---------|-----------|
| PostgreSQL | Structured data | DTC codes, users, diagnosis history |
| Neo4j | Graph relationships | DTC -> Symptom -> Component -> Repair |
| Qdrant | Vector search | Hungarian embeddings (768-dim huBERT) |

All three databases should be backed up together to ensure data consistency.

## Backup Script

Location: `scripts/backup_data.py`

### Basic Usage

```bash
# Full backup of all databases
python scripts/backup_data.py --all

# Backup specific databases
python scripts/backup_data.py --postgres
python scripts/backup_data.py --neo4j
python scripts/backup_data.py --qdrant
python scripts/backup_data.py --json

# Incremental backup (changes since last backup)
python scripts/backup_data.py --incremental

# Incremental since specific date
python scripts/backup_data.py --incremental --since 2024-01-01
```

### Backup Management

```bash
# List all available backups
python scripts/backup_data.py --list

# Verify backup integrity
python scripts/backup_data.py --verify full_20240201_120000

# Cleanup old backups (keep last 5)
python scripts/backup_data.py --cleanup --keep 5
```

### Restore from Backup

```bash
# Restore all databases from backup
python scripts/backup_data.py --restore full_20240201_120000

# Restore specific database only
python scripts/backup_data.py --restore full_20240201_120000 --target postgres

# Preview restore without making changes
python scripts/backup_data.py --restore full_20240201_120000 --dry-run
```

### Cloud Storage Upload

```bash
# Upload to AWS S3
python scripts/backup_data.py --all --upload-s3 my-backup-bucket

# Upload to S3 with custom prefix and region
python scripts/backup_data.py --all --upload-s3 my-bucket --s3-prefix autocognitix/backups --s3-region eu-central-1

# Upload to S3-compatible storage (MinIO, DigitalOcean Spaces)
python scripts/backup_data.py --all --upload-s3 my-bucket --s3-endpoint https://nyc3.digitaloceanspaces.com

# Upload to Google Cloud Storage
python scripts/backup_data.py --all --upload-gcs my-gcs-bucket

# Upload to Azure Blob Storage
python scripts/backup_data.py --all --upload-azure my-container

# List cloud backups
python scripts/backup_data.py --list-cloud s3 --cloud-bucket my-bucket --s3-prefix backups
```

### Environment Variables for Cloud Storage

```bash
# AWS S3
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=eu-central-1

# Google Cloud Storage
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Azure Blob Storage
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."
```

## Export Script

Location: `scripts/export_data.py`

### Export Formats

```bash
# Export to all formats
python scripts/export_data.py --all

# Export to specific formats
python scripts/export_data.py --csv              # CSV files
python scripts/export_data.py --excel            # Excel workbook (.xlsx)
python scripts/export_data.py --json             # JSON format
python scripts/export_data.py --sqlite           # SQLite database
python scripts/export_data.py --graphml          # Neo4j graph (GraphML)
python scripts/export_data.py --qdrant           # Qdrant vectors
python scripts/export_data.py --translations     # Translations only
python scripts/export_data.py --history          # Diagnosis history
```

### Filtering Options

```bash
# Filter by DTC category
python scripts/export_data.py --excel --category P          # Powertrain only
python scripts/export_data.py --excel --category P,B,C      # Multiple categories

# Filter by manufacturer
python scripts/export_data.py --csv --manufacturer Toyota

# Filter by translation status
python scripts/export_data.py --json --translated           # Only translated codes
python scripts/export_data.py --json --untranslated         # Only untranslated codes

# Filter by date range
python scripts/export_data.py --csv --since 2024-01-01 --until 2024-06-30

# Custom output directory
python scripts/export_data.py --all --output-dir /path/to/exports
```

### Export Output Structure

```
data/exports/20240201_120000/
├── manifest.json           # Export metadata
├── csv/
│   ├── pg_dtc_codes.csv
│   ├── neo4j_DTCNode.csv
│   └── dtc_codes_full.csv
├── autocognitix_export.xlsx
├── autocognitix_export.json
├── autocognitix_export.db   # SQLite
├── neo4j_graph.graphml
├── qdrant_vectors.json.gz
├── translations.json
└── diagnosis_history.json
```

## Import Script

Location: `scripts/import_data.py`

### Import from Different Formats

```bash
# Import from JSON export
python scripts/import_data.py --json path/to/export.json

# Import from CSV directory
python scripts/import_data.py --csv path/to/csv/directory

# Import from SQLite database
python scripts/import_data.py --sqlite path/to/export.db

# Import from GraphML (Neo4j)
python scripts/import_data.py --graphml path/to/graph.graphml
```

### Conflict Resolution Strategies

```bash
# Skip existing records (default)
python scripts/import_data.py --json export.json --on-conflict skip

# Overwrite existing records
python scripts/import_data.py --json export.json --on-conflict overwrite

# Merge fields (keep existing, add new)
python scripts/import_data.py --json export.json --on-conflict merge

# Keep newest (by timestamp)
python scripts/import_data.py --json export.json --on-conflict newest
```

### Validation and Preview

```bash
# Validate data without importing
python scripts/import_data.py --json export.json --validate

# Preview import without making changes
python scripts/import_data.py --json export.json --dry-run
```

## Data Sync Script

Location: `scripts/data_sync.py`

### Synchronization Operations

```bash
# Sync all databases
python scripts/data_sync.py --all

# Sync specific pairs
python scripts/data_sync.py --postgres-neo4j      # PostgreSQL -> Neo4j
python scripts/data_sync.py --postgres-qdrant     # PostgreSQL -> Qdrant
python scripts/data_sync.py --neo4j-postgres      # Neo4j -> PostgreSQL
python scripts/data_sync.py --translations        # Sync translations

# Preview sync without changes
python scripts/data_sync.py --all --dry-run

# Force sync (update existing records)
python scripts/data_sync.py --all --force

# Custom batch size for large syncs
python scripts/data_sync.py --all --batch-size 500
```

### Consistency Verification

```bash
# Verify data consistency
python scripts/data_sync.py --verify

# Save verification report to file
python scripts/data_sync.py --verify --report /path/to/report.json
```

## Cloud Storage Integration

### AWS S3 Setup

1. Create an S3 bucket for backups
2. Create an IAM user with S3 access
3. Set environment variables:

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_DEFAULT_REGION=eu-central-1
```

4. Install boto3:

```bash
pip install boto3
```

### Google Cloud Storage Setup

1. Create a GCS bucket
2. Create a service account with Storage Admin role
3. Download the service account JSON key
4. Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

5. Install google-cloud-storage:

```bash
pip install google-cloud-storage
```

### Azure Blob Storage Setup

1. Create a Storage Account
2. Get the connection string from Azure Portal
3. Set environment variable:

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=mystorageaccount;AccountKey=...;EndpointSuffix=core.windows.net"
```

4. Install azure-storage-blob:

```bash
pip install azure-storage-blob
```

### S3-Compatible Storage (MinIO, DigitalOcean Spaces)

```bash
# MinIO
python scripts/backup_data.py --all \
  --upload-s3 my-bucket \
  --s3-endpoint http://localhost:9000

# DigitalOcean Spaces
python scripts/backup_data.py --all \
  --upload-s3 my-space \
  --s3-endpoint https://nyc3.digitaloceanspaces.com \
  --s3-region nyc3
```

## Scheduled Backups (Cron)

### Daily Full Backup

```cron
# Daily backup at 2:00 AM
0 2 * * * cd /path/to/AutoCognitix && /path/to/venv/bin/python scripts/backup_data.py --all --upload-s3 my-backup-bucket >> /var/log/autocognitix/backup.log 2>&1
```

### Hourly Incremental Backup

```cron
# Incremental backup every hour
0 * * * * cd /path/to/AutoCognitix && /path/to/venv/bin/python scripts/backup_data.py --incremental >> /var/log/autocognitix/backup.log 2>&1
```

### Weekly Cleanup

```cron
# Keep last 30 backups, run weekly on Sunday at 3:00 AM
0 3 * * 0 cd /path/to/AutoCognitix && /path/to/venv/bin/python scripts/backup_data.py --cleanup --keep 30 >> /var/log/autocognitix/cleanup.log 2>&1
```

### Daily Data Verification

```cron
# Verify data consistency daily at 4:00 AM
0 4 * * * cd /path/to/AutoCognitix && /path/to/venv/bin/python scripts/data_sync.py --verify --report /var/log/autocognitix/verification-$(date +\%Y\%m\%d).json >> /var/log/autocognitix/verify.log 2>&1
```

### Complete Cron Example

```cron
# AutoCognitix Backup Schedule
# ============================

# Environment setup
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
MAILTO=admin@example.com
AUTOCOGNITIX_DIR=/path/to/AutoCognitix
VENV=/path/to/venv/bin/python
LOG_DIR=/var/log/autocognitix

# Daily full backup at 2:00 AM with S3 upload
0 2 * * * cd $AUTOCOGNITIX_DIR && $VENV scripts/backup_data.py --all --upload-s3 autocognitix-backups >> $LOG_DIR/backup.log 2>&1

# Hourly incremental backup (skip 2 AM when full runs)
0 0,1,3-23 * * * cd $AUTOCOGNITIX_DIR && $VENV scripts/backup_data.py --incremental >> $LOG_DIR/backup-incr.log 2>&1

# Weekly cleanup on Sunday at 3:00 AM
0 3 * * 0 cd $AUTOCOGNITIX_DIR && $VENV scripts/backup_data.py --cleanup --keep 30 >> $LOG_DIR/cleanup.log 2>&1

# Daily consistency check at 4:00 AM
0 4 * * * cd $AUTOCOGNITIX_DIR && $VENV scripts/data_sync.py --verify --report $LOG_DIR/verify-$(date +\%Y\%m\%d).json >> $LOG_DIR/verify.log 2>&1

# Weekly full sync on Saturday at 5:00 AM
0 5 * * 6 cd $AUTOCOGNITIX_DIR && $VENV scripts/data_sync.py --all --force >> $LOG_DIR/sync.log 2>&1
```

### Systemd Timer Alternative

Create `/etc/systemd/system/autocognitix-backup.service`:

```ini
[Unit]
Description=AutoCognitix Backup
After=network.target

[Service]
Type=oneshot
User=autocognitix
WorkingDirectory=/path/to/AutoCognitix
ExecStart=/path/to/venv/bin/python scripts/backup_data.py --all --upload-s3 autocognitix-backups
StandardOutput=append:/var/log/autocognitix/backup.log
StandardError=append:/var/log/autocognitix/backup.log
```

Create `/etc/systemd/system/autocognitix-backup.timer`:

```ini
[Unit]
Description=Daily AutoCognitix Backup

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable the timer:

```bash
sudo systemctl enable autocognitix-backup.timer
sudo systemctl start autocognitix-backup.timer
```

## Disaster Recovery

### Full Recovery Procedure

1. **Identify the Latest Good Backup**

```bash
# List local backups
python scripts/backup_data.py --list

# Or list cloud backups
python scripts/backup_data.py --list-cloud s3 --cloud-bucket autocognitix-backups
```

2. **Download from Cloud (if needed)**

```bash
# The backup script includes download functionality
# Or use AWS CLI / gsutil / az directly
aws s3 cp s3://autocognitix-backups/backups/full_20240201_120000.tar.gz ./
```

3. **Extract and Verify**

```bash
# Extract backup archive
tar -xzf full_20240201_120000.tar.gz

# Verify backup integrity
python scripts/backup_data.py --verify full_20240201_120000
```

4. **Restore Databases**

```bash
# Restore all databases
python scripts/backup_data.py --restore full_20240201_120000

# Or restore specific databases
python scripts/backup_data.py --restore full_20240201_120000 --target postgres
python scripts/backup_data.py --restore full_20240201_120000 --target neo4j
python scripts/backup_data.py --restore full_20240201_120000 --target qdrant
```

5. **Verify Data Consistency**

```bash
python scripts/data_sync.py --verify
```

6. **Sync if Needed**

```bash
python scripts/data_sync.py --all --force
```

### Partial Recovery

If only one database needs recovery:

```bash
# Restore PostgreSQL only
python scripts/backup_data.py --restore full_20240201_120000 --target postgres

# Sync to other databases
python scripts/data_sync.py --postgres-neo4j --postgres-qdrant --force
```

## Best Practices

### Backup Strategy

1. **3-2-1 Rule**: Keep 3 copies on 2 different media with 1 offsite
2. **Regular Testing**: Test restore procedures monthly
3. **Encryption**: Enable server-side encryption for cloud backups
4. **Retention Policy**: Keep daily backups for 7 days, weekly for 4 weeks, monthly for 12 months

### Security

1. **Credentials**: Never commit credentials to version control
2. **IAM Policies**: Use least-privilege access for backup accounts
3. **Encryption**: Enable encryption at rest for all cloud storage
4. **Access Logs**: Enable access logging for audit trails

### Monitoring

1. **Backup Alerts**: Set up alerts for backup failures
2. **Storage Monitoring**: Monitor backup storage usage
3. **Verification Reports**: Review consistency reports regularly
4. **Log Rotation**: Implement log rotation for backup logs

### Performance

1. **Off-Peak Hours**: Schedule backups during low-traffic periods
2. **Incremental Backups**: Use incremental backups for frequent backups
3. **Compression**: Backups are compressed by default (gzip)
4. **Batch Size**: Adjust batch size for large syncs

## Troubleshooting

### Common Issues

**Backup fails with "pg_dump not found"**
- The script will automatically fall back to Python-based export
- Install PostgreSQL client tools for better performance

**Neo4j connection refused**
- Check NEO4J_URI in environment
- Verify Neo4j is running

**Qdrant collection not found**
- Collection may not exist yet
- Run data sync to create collections

**S3 upload fails with credentials error**
- Verify AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
- Check IAM permissions

### Debug Mode

```bash
# Enable verbose logging
python scripts/backup_data.py --all --verbose
python scripts/data_sync.py --verify --verbose
```

### Getting Help

```bash
# Show help for any script
python scripts/backup_data.py --help
python scripts/export_data.py --help
python scripts/import_data.py --help
python scripts/data_sync.py --help
```
