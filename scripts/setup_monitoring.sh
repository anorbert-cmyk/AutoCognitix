#!/bin/bash
# ============================================
# AutoCognitix - Monitoring Stack Setup Script
# Sets up Prometheus, Grafana, and exporters
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
PROMETHEUS_VERSION="v2.48.0"
GRAFANA_VERSION="10.2.2"
ALERTMANAGER_VERSION="v0.26.0"
NODE_EXPORTER_VERSION="v1.7.0"

# Print functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    print_success "Docker daemon is running"
}

# Create required directories
create_directories() {
    print_header "Creating Required Directories"

    local dirs=(
        "$PROJECT_ROOT/docker/prometheus"
        "$PROJECT_ROOT/docker/prometheus/rules"
        "$PROJECT_ROOT/docker/grafana/provisioning/datasources"
        "$PROJECT_ROOT/docker/grafana/provisioning/dashboards"
        "$PROJECT_ROOT/docker/grafana/provisioning/dashboards/json"
        "$PROJECT_ROOT/docker/alertmanager"
        "$PROJECT_ROOT/data/prometheus"
        "$PROJECT_ROOT/data/grafana"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created: $dir"
        else
            print_info "Already exists: $dir"
        fi
    done
}

# Generate Alertmanager config
generate_alertmanager_config() {
    print_header "Generating Alertmanager Configuration"

    local config_file="$PROJECT_ROOT/docker/alertmanager/alertmanager.yml"

    if [ -f "$config_file" ]; then
        print_warning "Alertmanager config already exists, skipping..."
        return
    fi

    cat > "$config_file" << 'EOF'
# ============================================
# AutoCognitix - Alertmanager Configuration
# ============================================

global:
  resolve_timeout: 5m
  # SMTP settings for email notifications (configure as needed)
  # smtp_smarthost: 'smtp.example.com:587'
  # smtp_from: 'alertmanager@autocognitix.hu'
  # smtp_auth_username: 'alertmanager@autocognitix.hu'
  # smtp_auth_password: 'password'

# Route configuration
route:
  group_by: ['alertname', 'severity', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default-receiver'

  routes:
    # Critical alerts
    - match:
        severity: critical
      receiver: 'critical-receiver'
      group_wait: 10s
      repeat_interval: 1h

    # Warning alerts
    - match:
        severity: warning
      receiver: 'warning-receiver'
      group_wait: 1m
      repeat_interval: 4h

# Inhibition rules
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'service']

# Receivers
receivers:
  - name: 'default-receiver'
    # Configure webhook, email, Slack, etc.
    # webhook_configs:
    #   - url: 'http://localhost:5001/'

  - name: 'critical-receiver'
    # webhook_configs:
    #   - url: 'http://localhost:5001/critical'
    # email_configs:
    #   - to: 'oncall@autocognitix.hu'

  - name: 'warning-receiver'
    # webhook_configs:
    #   - url: 'http://localhost:5001/warning'
EOF

    print_success "Created Alertmanager configuration"
}

# Set up Grafana admin password
setup_grafana_credentials() {
    print_header "Setting Up Grafana Credentials"

    local env_file="$PROJECT_ROOT/.env.monitoring"

    if [ -f "$env_file" ]; then
        print_warning "Monitoring environment file already exists"
        return
    fi

    # Generate random password
    local grafana_password=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)

    cat > "$env_file" << EOF
# AutoCognitix Monitoring Environment Variables
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=${grafana_password}
GRAFANA_SECRET_KEY=$(openssl rand -hex 32)

# Prometheus
PROMETHEUS_RETENTION_TIME=15d
PROMETHEUS_RETENTION_SIZE=10GB

# Alertmanager
ALERTMANAGER_SLACK_WEBHOOK=
ALERTMANAGER_EMAIL_TO=
EOF

    chmod 600 "$env_file"

    print_success "Created monitoring environment file"
    print_info "Grafana admin password: ${grafana_password}"
    print_warning "IMPORTANT: Save this password! It's stored in .env.monitoring"
}

# Validate Prometheus configuration
validate_prometheus_config() {
    print_header "Validating Prometheus Configuration"

    local config_file="$PROJECT_ROOT/docker/prometheus/prometheus.yml"

    if [ ! -f "$config_file" ]; then
        print_error "Prometheus configuration not found at: $config_file"
        return 1
    fi

    # Use promtool to validate if available
    if docker run --rm -v "$PROJECT_ROOT/docker/prometheus:/prometheus" \
        prom/prometheus:$PROMETHEUS_VERSION promtool check config /prometheus/prometheus.yml 2>/dev/null; then
        print_success "Prometheus configuration is valid"
    else
        print_warning "Could not validate Prometheus config (promtool not available or config has issues)"
    fi
}

# Start monitoring stack
start_monitoring_stack() {
    print_header "Starting Monitoring Stack"

    cd "$PROJECT_ROOT"

    # Check if monitoring compose file exists
    if [ ! -f "docker-compose.monitoring.yml" ]; then
        print_error "docker-compose.monitoring.yml not found"
        print_info "Please run this script after the monitoring compose file has been created"
        return 1
    fi

    # Start the monitoring stack
    print_info "Starting monitoring services..."

    if docker compose -f docker-compose.monitoring.yml up -d; then
        print_success "Monitoring stack started successfully"
    else
        print_error "Failed to start monitoring stack"
        return 1
    fi

    # Wait for services to be healthy
    print_info "Waiting for services to be ready..."
    sleep 10

    # Check service status
    print_header "Service Status"
    docker compose -f docker-compose.monitoring.yml ps
}

# Stop monitoring stack
stop_monitoring_stack() {
    print_header "Stopping Monitoring Stack"

    cd "$PROJECT_ROOT"

    if [ -f "docker-compose.monitoring.yml" ]; then
        docker compose -f docker-compose.monitoring.yml down
        print_success "Monitoring stack stopped"
    else
        print_warning "docker-compose.monitoring.yml not found"
    fi
}

# Show access information
show_access_info() {
    print_header "Access Information"

    echo -e "
${GREEN}Monitoring Services:${NC}

  ${BLUE}Prometheus:${NC}     http://localhost:9090
  ${BLUE}Grafana:${NC}        http://localhost:3001
  ${BLUE}Alertmanager:${NC}   http://localhost:9093

${GREEN}Default Credentials:${NC}

  ${BLUE}Grafana:${NC}
    Username: admin
    Password: See .env.monitoring file

${GREEN}Useful Commands:${NC}

  Start monitoring:   docker compose -f docker-compose.monitoring.yml up -d
  Stop monitoring:    docker compose -f docker-compose.monitoring.yml down
  View logs:          docker compose -f docker-compose.monitoring.yml logs -f
  Prometheus reload:  curl -X POST http://localhost:9090/-/reload

${GREEN}Documentation:${NC}

  Prometheus:    https://prometheus.io/docs/
  Grafana:       https://grafana.com/docs/
  Alertmanager:  https://prometheus.io/docs/alerting/latest/alertmanager/
"
}

# Health check
health_check() {
    print_header "Running Health Checks"

    local services=("prometheus:9090" "grafana:3001" "alertmanager:9093")
    local all_healthy=true

    for service in "${services[@]}"; do
        local name=$(echo $service | cut -d: -f1)
        local port=$(echo $service | cut -d: -f2)

        if curl -s "http://localhost:$port/-/healthy" > /dev/null 2>&1 || \
           curl -s "http://localhost:$port/api/health" > /dev/null 2>&1 || \
           curl -s "http://localhost:$port/" > /dev/null 2>&1; then
            print_success "$name is healthy (port $port)"
        else
            print_error "$name is not responding (port $port)"
            all_healthy=false
        fi
    done

    if $all_healthy; then
        print_success "All services are healthy!"
        return 0
    else
        print_error "Some services are not healthy"
        return 1
    fi
}

# Main menu
show_menu() {
    echo -e "
${BLUE}AutoCognitix Monitoring Setup${NC}

Usage: $0 [command]

Commands:
    setup       Full setup (creates directories, configs, starts services)
    start       Start the monitoring stack
    stop        Stop the monitoring stack
    restart     Restart the monitoring stack
    status      Show service status
    health      Run health checks
    validate    Validate Prometheus configuration
    info        Show access information
    help        Show this help message
"
}

# Main function
main() {
    local command=${1:-help}

    case $command in
        setup)
            check_prerequisites
            create_directories
            generate_alertmanager_config
            setup_grafana_credentials
            validate_prometheus_config
            start_monitoring_stack
            show_access_info
            ;;
        start)
            start_monitoring_stack
            show_access_info
            ;;
        stop)
            stop_monitoring_stack
            ;;
        restart)
            stop_monitoring_stack
            sleep 2
            start_monitoring_stack
            ;;
        status)
            print_header "Service Status"
            cd "$PROJECT_ROOT"
            docker compose -f docker-compose.monitoring.yml ps 2>/dev/null || \
                print_warning "Monitoring stack not running"
            ;;
        health)
            health_check
            ;;
        validate)
            validate_prometheus_config
            ;;
        info)
            show_access_info
            ;;
        help|--help|-h)
            show_menu
            ;;
        *)
            print_error "Unknown command: $command"
            show_menu
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
