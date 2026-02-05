#!/usr/bin/env bash
# ============================================
# AutoCognitix - Production Deployment Script
# ============================================
# Usage:
#   ./deploy.sh [command] [options]
#
# Commands:
#   deploy      - Full deployment (default)
#   build       - Build images only
#   migrate     - Run database migrations
#   rollback    - Rollback to previous version
#   health      - Check health of all services
#   logs        - View logs
#   backup      - Create database backup
#   restore     - Restore from backup
#   status      - Show service status
#
# Options:
#   --tag       - Image tag to deploy (default: latest)
#   --env       - Environment file (default: .env.production)
#   --no-cache  - Build without cache
#   --force     - Force deployment without health checks
# ============================================

set -euo pipefail

# ==========================================
# Configuration
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENV_FILE="${ENV_FILE:-.env.production}"
NO_CACHE=""
FORCE=false
MAX_RETRIES=30
RETRY_INTERVAL=10

# ==========================================
# Utility Functions
# ==========================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

die() {
    log_error "$*"
    exit 1
}

confirm() {
    local prompt="${1:-Are you sure?}"
    read -r -p "${prompt} [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# ==========================================
# Docker Compose Wrapper
# ==========================================
dc() {
    docker compose -f "${COMPOSE_FILE}" --env-file "${PROJECT_ROOT}/${ENV_FILE}" "$@"
}

# ==========================================
# Pre-flight Checks
# ==========================================
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        die "Docker is not installed"
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        die "Docker Compose v2 is not installed"
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        die "Docker daemon is not running"
    fi

    # Check environment file
    if [[ ! -f "${PROJECT_ROOT}/${ENV_FILE}" ]]; then
        die "Environment file not found: ${PROJECT_ROOT}/${ENV_FILE}"
    fi

    # Check compose file
    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        die "Compose file not found: ${COMPOSE_FILE}"
    fi

    log_success "Prerequisites check passed"
}

# ==========================================
# Data Directory Setup
# ==========================================
setup_data_dirs() {
    log_info "Setting up data directories..."

    local data_path="${DATA_PATH:-${PROJECT_ROOT}/data}"
    local dirs=(
        "${data_path}/postgres"
        "${data_path}/backups/postgres"
        "${data_path}/neo4j/data"
        "${data_path}/neo4j/logs"
        "${data_path}/neo4j/plugins"
        "${data_path}/qdrant/storage"
        "${data_path}/qdrant/snapshots"
        "${data_path}/redis"
        "${data_path}/traefik/letsencrypt"
        "${data_path}/traefik/logs"
    )

    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done

    # Set proper permissions for Let's Encrypt
    chmod 600 "${data_path}/traefik/letsencrypt" 2>/dev/null || true

    log_success "Data directories ready"
}

# ==========================================
# Build Functions
# ==========================================
build_images() {
    log_info "Building Docker images with tag: ${IMAGE_TAG}"

    local build_args=(
        --build-arg "VITE_APP_VERSION=${IMAGE_TAG}"
    )

    if [[ -n "${NO_CACHE}" ]]; then
        build_args+=(--no-cache)
    fi

    # Build backend
    log_info "Building backend image..."
    dc build ${build_args[@]} backend || die "Failed to build backend"

    # Build frontend
    log_info "Building frontend image..."
    dc build ${build_args[@]} frontend || die "Failed to build frontend"

    log_success "Images built successfully"
}

# ==========================================
# Migration Functions
# ==========================================
run_migrations() {
    log_info "Running database migrations..."

    # Ensure PostgreSQL is healthy
    wait_for_service "postgres" 60

    # Run migrations using the migration profile
    dc run --rm migration || die "Migration failed"

    log_success "Migrations completed"
}

# ==========================================
# Health Check Functions
# ==========================================
wait_for_service() {
    local service="$1"
    local max_retries="${2:-$MAX_RETRIES}"
    local retry=0

    log_info "Waiting for ${service} to be healthy..."

    while [[ $retry -lt $max_retries ]]; do
        if dc ps "${service}" 2>/dev/null | grep -q "healthy"; then
            log_success "${service} is healthy"
            return 0
        fi

        retry=$((retry + 1))
        log_info "Waiting for ${service}... (${retry}/${max_retries})"
        sleep "${RETRY_INTERVAL}"
    done

    log_error "${service} health check failed after ${max_retries} attempts"
    return 1
}

check_all_services_health() {
    log_info "Checking health of all services..."

    local services=("postgres" "neo4j" "qdrant" "redis" "backend" "frontend")
    local failed=()

    for service in "${services[@]}"; do
        if ! wait_for_service "$service" 10; then
            failed+=("$service")
        fi
    done

    if [[ ${#failed[@]} -gt 0 ]]; then
        log_error "The following services are unhealthy: ${failed[*]}"
        return 1
    fi

    log_success "All services are healthy"
    return 0
}

health_check() {
    log_info "Performing health check..."

    # Check Traefik
    if dc ps traefik 2>/dev/null | grep -q "running"; then
        log_success "Traefik: running"
    else
        log_warning "Traefik: not running"
    fi

    # Check all services
    for service in postgres neo4j qdrant redis backend frontend; do
        local status
        status=$(dc ps "${service}" 2>/dev/null | tail -n 1 | awk '{print $NF}' || echo "unknown")
        if [[ "$status" == *"healthy"* ]]; then
            log_success "${service}: healthy"
        elif [[ "$status" == *"running"* ]]; then
            log_warning "${service}: running (no health status)"
        else
            log_error "${service}: ${status}"
        fi
    done

    # Check endpoints
    log_info "Checking API endpoints..."

    local domain
    domain=$(grep "^DOMAIN=" "${PROJECT_ROOT}/${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' || echo "localhost")

    # Backend health
    if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
        log_success "Backend API: responsive"
    else
        log_warning "Backend API: not responsive (may be behind proxy)"
    fi

    # Frontend health
    if curl -sf "http://localhost:3000/health" >/dev/null 2>&1; then
        log_success "Frontend: responsive"
    else
        log_warning "Frontend: not responsive (may be behind proxy)"
    fi
}

# ==========================================
# Deployment Functions
# ==========================================
deploy() {
    log_info "Starting deployment..."

    check_prerequisites
    setup_data_dirs

    # Store current image tags for rollback
    save_current_state

    # Build images
    build_images

    # Pull base images
    log_info "Pulling base images..."
    dc pull --ignore-buildable || log_warning "Some images could not be pulled"

    # Start databases first
    log_info "Starting database services..."
    dc up -d postgres neo4j qdrant redis

    # Wait for databases to be healthy
    wait_for_service "postgres" 60 || die "PostgreSQL failed to start"
    wait_for_service "neo4j" 90 || die "Neo4j failed to start"
    wait_for_service "qdrant" 60 || die "Qdrant failed to start"
    wait_for_service "redis" 30 || die "Redis failed to start"

    # Run migrations
    run_migrations

    # Start application services
    log_info "Starting application services..."
    dc up -d traefik backend frontend

    # Wait for application services
    if [[ "${FORCE}" != "true" ]]; then
        wait_for_service "backend" 90 || die "Backend failed to start"
        wait_for_service "frontend" 30 || die "Frontend failed to start"
    fi

    # Final health check
    health_check

    log_success "Deployment completed successfully!"
    log_info "Application is available at: https://${DOMAIN:-localhost}"
}

# ==========================================
# Rollback Functions
# ==========================================
save_current_state() {
    local state_file="${PROJECT_ROOT}/.deploy-state"

    log_info "Saving current state for rollback..."

    # Get current image digests
    local backend_digest frontend_digest
    backend_digest=$(docker images autocognitix-backend:latest --format "{{.Digest}}" 2>/dev/null || echo "none")
    frontend_digest=$(docker images autocognitix-frontend:latest --format "{{.Digest}}" 2>/dev/null || echo "none")

    cat > "${state_file}" <<EOF
ROLLBACK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKEND_DIGEST=${backend_digest}
FRONTEND_DIGEST=${frontend_digest}
EOF

    log_success "State saved to ${state_file}"
}

rollback() {
    local state_file="${PROJECT_ROOT}/.deploy-state"

    if [[ ! -f "${state_file}" ]]; then
        die "No rollback state found. Cannot rollback."
    fi

    log_warning "Rolling back to previous deployment..."

    # Source the state file
    source "${state_file}"

    log_info "Rollback timestamp: ${ROLLBACK_TIMESTAMP:-unknown}"

    # Stop current services
    log_info "Stopping current services..."
    dc stop backend frontend

    # Restore previous images if available
    if [[ "${BACKEND_DIGEST:-none}" != "none" ]]; then
        log_info "Restoring backend image..."
        docker tag "autocognitix-backend@${BACKEND_DIGEST}" "autocognitix-backend:rollback" 2>/dev/null || \
            log_warning "Could not restore backend image"
    fi

    if [[ "${FRONTEND_DIGEST:-none}" != "none" ]]; then
        log_info "Restoring frontend image..."
        docker tag "autocognitix-frontend@${FRONTEND_DIGEST}" "autocognitix-frontend:rollback" 2>/dev/null || \
            log_warning "Could not restore frontend image"
    fi

    # Start services with rollback tag
    log_info "Starting services with rollback images..."
    IMAGE_TAG=rollback dc up -d backend frontend

    # Health check
    if ! check_all_services_health; then
        log_error "Rollback failed! Manual intervention required."
        exit 1
    fi

    log_success "Rollback completed successfully"
}

# ==========================================
# Backup Functions
# ==========================================
backup() {
    local backup_dir="${PROJECT_ROOT}/data/backups"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)

    log_info "Creating backup..."

    mkdir -p "${backup_dir}"

    # PostgreSQL backup
    log_info "Backing up PostgreSQL..."
    dc exec -T postgres pg_dumpall -U "${POSTGRES_USER:-autocognitix}" | \
        gzip > "${backup_dir}/postgres_${timestamp}.sql.gz" || \
        log_warning "PostgreSQL backup failed"

    # Neo4j backup (using APOC)
    log_info "Backing up Neo4j..."
    dc exec -T neo4j neo4j-admin database dump neo4j --to-path=/backups 2>/dev/null || \
        log_warning "Neo4j backup failed (may require enterprise edition)"

    # Qdrant snapshot
    log_info "Creating Qdrant snapshot..."
    curl -sf -X POST "http://localhost:6333/collections/dtc_codes/snapshots" > /dev/null 2>&1 || \
        log_warning "Qdrant snapshot failed"

    # Redis backup (copy RDB file)
    log_info "Backing up Redis..."
    dc exec -T redis redis-cli BGSAVE > /dev/null 2>&1
    sleep 5
    cp "${PROJECT_ROOT}/data/redis/dump.rdb" "${backup_dir}/redis_${timestamp}.rdb" 2>/dev/null || \
        log_warning "Redis backup failed"

    log_success "Backup completed: ${backup_dir}/*_${timestamp}.*"
}

restore() {
    local backup_file="$1"

    if [[ -z "${backup_file}" ]]; then
        log_error "Usage: $0 restore <backup_file>"
        exit 1
    fi

    if [[ ! -f "${backup_file}" ]]; then
        die "Backup file not found: ${backup_file}"
    fi

    log_warning "This will restore from backup: ${backup_file}"
    confirm "Continue?" || exit 0

    # Determine backup type and restore
    case "${backup_file}" in
        *.sql.gz)
            log_info "Restoring PostgreSQL from ${backup_file}..."
            gunzip -c "${backup_file}" | dc exec -T postgres psql -U "${POSTGRES_USER:-autocognitix}"
            ;;
        *.rdb)
            log_info "Restoring Redis from ${backup_file}..."
            dc stop redis
            cp "${backup_file}" "${PROJECT_ROOT}/data/redis/dump.rdb"
            dc start redis
            ;;
        *)
            die "Unknown backup format: ${backup_file}"
            ;;
    esac

    log_success "Restore completed"
}

# ==========================================
# Log Functions
# ==========================================
show_logs() {
    local service="${1:-}"
    local tail="${2:-100}"

    if [[ -n "${service}" ]]; then
        dc logs -f --tail="${tail}" "${service}"
    else
        dc logs -f --tail="${tail}"
    fi
}

# ==========================================
# Status Functions
# ==========================================
show_status() {
    log_info "Service Status:"
    echo ""
    dc ps
    echo ""

    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" \
        $(dc ps -q) 2>/dev/null || true
}

# ==========================================
# Cleanup Functions
# ==========================================
cleanup() {
    log_info "Cleaning up unused Docker resources..."

    confirm "This will remove unused images, volumes, and networks. Continue?" || exit 0

    # Remove stopped containers
    docker container prune -f

    # Remove unused images
    docker image prune -f

    # Remove unused volumes (be careful!)
    # docker volume prune -f

    # Remove unused networks
    docker network prune -f

    log_success "Cleanup completed"
}

# ==========================================
# Stop/Down Functions
# ==========================================
stop() {
    log_info "Stopping services..."
    dc stop
    log_success "Services stopped"
}

down() {
    log_warning "This will stop and remove all containers (data will be preserved)"
    confirm "Continue?" || exit 0

    log_info "Stopping and removing containers..."
    dc down
    log_success "Services removed"
}

# ==========================================
# Parse Arguments
# ==========================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --env)
                ENV_FILE="$2"
                shift 2
                ;;
            --no-cache)
                NO_CACHE="--no-cache"
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            -*)
                die "Unknown option: $1"
                ;;
            *)
                break
                ;;
        esac
    done

    COMMAND="${1:-deploy}"
    shift || true
    ARGS=("$@")
}

# ==========================================
# Main
# ==========================================
main() {
    parse_args "$@"

    # Export variables for docker-compose
    export IMAGE_TAG
    export DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data}"

    case "${COMMAND}" in
        deploy)
            deploy
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        migrate)
            check_prerequisites
            run_migrations
            ;;
        rollback)
            check_prerequisites
            rollback
            ;;
        health)
            health_check
            ;;
        logs)
            show_logs "${ARGS[@]:-}"
            ;;
        backup)
            backup
            ;;
        restore)
            restore "${ARGS[0]:-}"
            ;;
        status)
            show_status
            ;;
        stop)
            stop
            ;;
        down)
            down
            ;;
        cleanup)
            cleanup
            ;;
        *)
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  deploy      Full deployment (default)"
            echo "  build       Build images only"
            echo "  migrate     Run database migrations"
            echo "  rollback    Rollback to previous version"
            echo "  health      Check health of all services"
            echo "  logs        View logs (optionally specify service)"
            echo "  backup      Create database backup"
            echo "  restore     Restore from backup"
            echo "  status      Show service status"
            echo "  stop        Stop all services"
            echo "  down        Stop and remove containers"
            echo "  cleanup     Clean unused Docker resources"
            echo ""
            echo "Options:"
            echo "  --tag TAG       Image tag (default: latest)"
            echo "  --env FILE      Environment file (default: .env.production)"
            echo "  --no-cache      Build without Docker cache"
            echo "  --force         Force deployment without health checks"
            exit 1
            ;;
    esac
}

main "$@"
