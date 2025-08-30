#!/bin/bash

# ====================
# FÁZE 1: ORCHESTRAČNÍ SKRIPT PRO AUTONOMNÍ PLATFORMU
# ====================

set -e

# Barvy pro výstup
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funkce pro logování
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Kontrola požadavků
check_requirements() {
    log_info "Kontrola systémových požadavků..."

    # Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker není nainstalován"
        exit 1
    fi

    # Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose není nainstalován"
        exit 1
    fi

    # Python 3.11+
    if ! python3 --version | grep -E "3\.(11|12)" &> /dev/null; then
        log_warning "Doporučená verze Python 3.11+, detekována: $(python3 --version)"
    fi

    log_success "Systémové požadavky splněny"
}

# Inicializace prostředí
init_environment() {
    log_info "Inicializace prostředí..."

    # Vytvoření .env souboru pokud neexistuje
    if [ ! -f .env ]; then
        if [ -f .env.template ]; then
            cp .env.template .env
            log_info "Vytvořen .env soubor z template"
            log_warning "DŮLEŽITÉ: Upravte .env soubor s vašimi API klíči!"
        else
            log_error ".env.template nenalezen"
            exit 1
        fi
    fi

    # Vytvoření potřebných adresářů
    mkdir -p data/parquet
    mkdir -p models
    mkdir -p research_cache
    mkdir -p artifacts
    mkdir -p logs

    log_success "Prostředí inicializováno"
}

# Validace před spuštěním
validate_setup() {
    log_info "Spouštění validace Fáze 1..."

    # Spuštění validačního skriptu
    if python3 scripts/validate_phase1.py; then
        log_success "Validace úspěšná"
    else
        log_error "Validace selhala"
        exit 1
    fi
}

# Spuštění služeb
start_services() {
    log_info "Spouštění Docker služeb..."

    # Sestavení obrazů
    docker-compose -f docker-compose.autonomous.yml build

    # Spuštění služeb
    docker-compose -f docker-compose.autonomous.yml up -d

    log_success "Služby spuštěny"

    # Čekání na inicializaci
    log_info "Čekání na inicializaci služeb..."
    sleep 30

    # Kontrola zdraví služeb
    check_services_health
}

# Kontrola zdraví služeb
check_services_health() {
    log_info "Kontrola zdraví služeb..."

    services=(
        "autonomous_scraper:8000:/health"
        "autonomous_milvus:9091:/healthz"
        "autonomous_prometheus:9090/-/healthy"
    )

    for service in "${services[@]}"; do
        IFS=':' read -r name port endpoint <<< "$service"

        if curl -f -s "http://localhost:${port}${endpoint}" > /dev/null; then
            log_success "Služba ${name} je zdravá"
        else
            log_warning "Služba ${name} není dostupná na portu ${port}"
        fi
    done
}

# Instalace Python závislostí
install_dependencies() {
    log_info "Instalace Python závislostí..."

    # Kontrola virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        log_warning "Není aktivováno virtuální prostředí"
        log_info "Vytváření virtuálního prostředí..."
        python3 -m venv venv
        source venv/bin/activate
    fi

    # Upgrade pip
    pip install --upgrade pip

    # Instalace závislostí s Metal podporou pro Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        log_info "Detekován Apple Silicon - aktivace Metal optimalizace"
        export CMAKE_ARGS="-DLLAMA_METAL=on"
        export FORCE_CMAKE=1
    fi

    pip install -r requirements.txt

    log_success "Závislosti nainstalovány"
}

# Stažení doporučeného LLM modelu
download_model() {
    log_info "Kontrola dostupnosti LLM modelu..."

    MODEL_DIR="./models"
    MODEL_NAME="mistral-7b-instruct-q4_k_m.gguf"
    MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

    if [ ! -f "$MODEL_PATH" ]; then
        log_info "LLM model nenalezen, stahování..."
        log_warning "Model má ~4GB, stahování může trvat několik minut"

        mkdir -p "$MODEL_DIR"

        # URL pro Mistral 7B model
        MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_k_m.gguf"

        if curl -L -o "$MODEL_PATH" "$MODEL_URL"; then
            log_success "Model stažen: $MODEL_PATH"
        else
            log_warning "Stahování modelu selhalo, pokračujeme bez lokálního LLM"
        fi
    else
        log_success "LLM model nalezen: $MODEL_PATH"
    fi

    # Aktualizace .env s cestou k modelu
    if [ -f "$MODEL_PATH" ]; then
        if grep -q "LLM_MODEL_PATH" .env; then
            sed -i.bak "s|LLM_MODEL_PATH=.*|LLM_MODEL_PATH=$MODEL_PATH|" .env
        else
            echo "LLM_MODEL_PATH=$MODEL_PATH" >> .env
        fi
    fi
}

# Zobrazení informací o běžících službách
show_services_info() {
    echo
    log_success "=== FÁZE 1 AUTONOMNÍ PLATFORMA SPUŠTĚNA ==="
    echo
    echo "Dostupné služby:"
    echo "• Hlavní API:           http://localhost:8000"
    echo "• API dokumentace:      http://localhost:8000/docs"
    echo "• Health check:         http://localhost:8000/health"
    echo "• Prometheus monitoring: http://localhost:9090"
    echo "• DuckDB web:           http://localhost:8080"
    echo
    echo "Příklady použití:"
    echo "• curl http://localhost:8000/health"
    echo "• curl http://localhost:8000/stats"
    echo
    echo "Pro zastavení služeb:"
    echo "• docker-compose -f docker-compose.autonomous.yml down"
    echo
}

# Vyčištění prostředí
cleanup() {
    log_info "Vyčištění prostředí..."

    # Zastavení služeb
    docker-compose -f docker-compose.autonomous.yml down

    # Vyčištění Docker volumes (optional)
    if [ "$1" = "--full" ]; then
        docker-compose -f docker-compose.autonomous.yml down -v
        docker system prune -f
        log_info "Kompletní vyčištění provedeno"
    fi

    log_success "Vyčištění dokončeno"
}

# Zobrazení nápovědy
show_help() {
    echo "Autonomní Research Platform - Fáze 1"
    echo
    echo "Použití: $0 [PŘÍKAZ]"
    echo
    echo "Příkazy:"
    echo "  start       Spustí kompletní platformu"
    echo "  stop        Zastaví všechny služby"
    echo "  restart     Restartuje platformu"
    echo "  status      Zobrazí stav služeb"
    echo "  validate    Spustí validace"
    echo "  logs        Zobrazí logy služeb"
    echo "  cleanup     Vyčistí prostředí"
    echo "  help        Zobrazí tuto nápovědu"
    echo
    echo "Možnosti:"
    echo "  --full      S cleanup provede kompletní vyčištění"
    echo
}

# Hlavní logika
main() {
    case "$1" in
        "start")
            check_requirements
            init_environment
            install_dependencies
            download_model
            validate_setup
            start_services
            show_services_info
            ;;
        "stop")
            cleanup
            ;;
        "restart")
            cleanup
            main start
            ;;
        "status")
            check_services_health
            ;;
        "validate")
            validate_setup
            ;;
        "logs")
            docker-compose -f docker-compose.autonomous.yml logs -f
            ;;
        "cleanup")
            cleanup $2
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        "")
            show_help
            ;;
        *)
            log_error "Neznámý příkaz: $1"
            show_help
            exit 1
            ;;
    esac
}

# Spuštění hlavní funkce
main "$@"
