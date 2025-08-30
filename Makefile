# DeepResearchTool - M1 Optimized Makefile
# Automatizace běžných úloh pro development optimalizovaný pro MacBook Air M1 (8GB RAM)

.PHONY: help setup lint format test build-docker run-local clean
.DEFAULT_GOAL := help

# ========================================
# HELP TARGET
# ========================================

help: ## Zobrazit dostupné příkazy
	@echo "DeepResearchTool - M1 Optimized Commands"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ========================================
# SETUP & INSTALLATION
# ========================================

setup: ## Instalace všech závislostí pro vývoj
	@echo "🚀 Nastavuji vývojové prostředí..."
	pip install --upgrade pip uv
	uv pip install -e ".[dev,m1-optimized]"
	playwright install chromium
	@echo "✅ Vývojové prostředí je připraveno!"

setup-minimal: ## Instalace pouze základních závislostí
	@echo "🚀 Instaluji základní závislosti..."
	pip install --upgrade pip uv
	uv pip install -e .
	@echo "✅ Základní závislosti nainstalovány!"

# ========================================
# CODE QUALITY
# ========================================

lint: ## Spuštění linteru (ruff check --fix)
	@echo "🔍 Spouštím linting..."
	ruff check --fix src/ tests/
	@echo "✅ Linting dokončen!"

format: ## Automatické formátování kódu (ruff format)
	@echo "✨ Formátuji kód..."
	ruff format src/ tests/
	@echo "✅ Formátování dokončeno!"

type-check: ## Spuštění type checkingu (mypy)
	@echo "🔎 Kontroluji typy..."
	mypy src/ --config-file pyproject.toml
	@echo "✅ Type checking dokončen!"

security: ## Bezpečnostní kontrola (bandit)
	@echo "🔒 Spouštím bezpečnostní scan..."
	bandit -r src/ -f txt
	@echo "✅ Bezpečnostní kontrola dokončena!"

# ========================================
# TESTING
# ========================================

test: ## Spuštění všech pytest testů
	@echo "🧪 Spouštím testy..."
	pytest tests/ -v --tb=short -m "not slow and not external"
	@echo "✅ Testy dokončeny!"

test-unit: ## Spuštění pouze unit testů
	@echo "🧪 Spouštím unit testy..."
	pytest tests/unit/ -v --tb=short
	@echo "✅ Unit testy dokončeny!"

test-integration: ## Spuštění integration testů
	@echo "🧪 Spouštím integration testy..."
	pytest tests/integration/ -v --tb=short
	@echo "✅ Integration testy dokončeny!"

test-coverage: ## Spuštění testů s pokrytím
	@echo "🧪 Spouštím testy s coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-fail-under=70
	@echo "✅ Coverage report vygenerován do htmlcov/"

# ========================================
# DOCKER OPERATIONS
# ========================================

build-docker: ## Sestavení všech Docker obrazů
	@echo "🐳 Sestavuji Docker obrazy..."
	docker build -t deepresearchtool:latest .
	docker build -f docker/Dockerfile.m1 -t deepresearchtool:m1 .
	@echo "✅ Docker obrazy sestaveny!"

build-docker-prod: ## Sestavení produkčního Docker obrazu
	@echo "🐳 Sestavuji produkční Docker obraz..."
	docker build -f docker/Dockerfile.production -t deepresearchtool:production .
	@echo "✅ Produkční Docker obraz sestaven!"

# ========================================
# LOCAL DEVELOPMENT (M1 OPTIMIZED)
# ========================================

run-local: ## Spuštění odlehčené verze aplikace pro lokální vývoj
	@echo "🚀 Spouštím lokální M1-optimalizovanou verzi..."
	@echo "⚡ Používám ChromaDB (in-process) místo Qdrant"
	@echo "💾 Omezená paměťová stopa pro M1 (8GB RAM)"
	docker-compose -f docker-compose.m1.yml up -d
	@echo "✅ Lokální služby spuštěny!"
	@echo "📊 Neo4j: http://localhost:7474"
	@echo "🔍 Redis: localhost:6379"
	@echo "🎯 Aplikace: http://localhost:8000"

stop-local: ## Zastavení lokálních služeb
	@echo "🛑 Zastavuji lokální služby..."
	docker-compose -f docker-compose.m1.yml down
	@echo "✅ Lokální služby zastaveny!"

logs-local: ## Zobrazení logů lokálních služeb
	docker-compose -f docker-compose.m1.yml logs -f

# ========================================
# DEVELOPMENT HELPERS
# ========================================

dev-server: ## Spuštění development serveru
	@echo "🎯 Spouštím development server..."
	python main.py --debug --reload

streamlit: ## Spuštění Streamlit dashboardu
	@echo "📊 Spouštím Streamlit dashboard..."
	streamlit run streamlit_dashboard.py

shell: ## Interaktivní Python shell s načtenými moduly
	@echo "🐍 Spouštím interaktivní shell..."
	python -i -c "from src.core.config import settings; print('⚡ DeepResearchTool shell ready!')"

# ========================================
# DATABASE OPERATIONS
# ========================================

db-setup: ## Nastavení databází (ChromaDB + Neo4j)
	@echo "🗄️ Nastavuji databáze..."
	python scripts/setup_databases.py
	@echo "✅ Databáze nastaveny!"

db-reset: ## Reset všech databází
	@echo "⚠️ Resetuji všechny databáze..."
	rm -rf ./chroma_db/
	docker-compose -f docker-compose.m1.yml exec neo4j cypher-shell "MATCH (n) DETACH DELETE n"
	@echo "✅ Databáze resetovány!"

# ========================================
# MONITORING & PERFORMANCE
# ========================================

monitor-memory: ## Sledování paměťové stopy (M1 optimized)
	@echo "📊 Sledování paměťové stopy..."
	@echo "Docker containers:"
	docker stats --no-stream
	@echo "\nPython processes:"
	ps aux | grep python | grep -v grep

benchmark: ## Performance benchmark pro M1
	@echo "⚡ Spouštím M1 benchmark..."
	python scripts/bench_m1_performance.py

# ========================================
# CLEANUP
# ========================================

clean: ## Vyčištění cache a dočasných souborů
	@echo "🧹 Čistím cache a dočasné soubory..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -f .coverage
	@echo "✅ Cleanup dokončen!"

clean-docker: ## Vyčištění Docker cache
	@echo "🐳 Čistím Docker cache..."
	docker system prune -f
	docker volume prune -f
	@echo "✅ Docker cache vyčištěn!"

clean-all: clean clean-docker ## Kompletní vyčištění
	@echo "🧹 Kompletní cleanup dokončen!"

# ========================================
# VALIDATION
# ========================================

validate: lint type-check security test ## Kompletní validace projektu
	@echo "✅ Projekt úspěšně validován!"

ci-check: setup-minimal validate ## CI/CD kontrola
	@echo "🎉 CI/CD kontrola úspěšná!"

# ========================================
# QUICK START
# ========================================

quick-start: setup run-local ## Rychlé spuštění pro nové vývojáře
	@echo "🎊 Rychlé spuštění dokončeno!"
	@echo ""
	@echo "Další kroky:"
	@echo "1. Zkopírujte .env.example do .env a vyplňte API klíče"
	@echo "2. Spusťte 'make dev-server' pro development server"
	@echo "3. Spusťte 'make streamlit' pro dashboard"
	@echo "4. Spusťte 'make test' pro ověření funkčnosti"

# ========================================
# PHASE IMPLEMENTATIONS
# ========================================

phase1-test: ## Test implementace Fáze 1 (stabilizace)
	@echo "🧪 Testování Fáze 1 - Stabilizace..."
	python -m pytest tests/phase1/ -v
	@echo "✅ Fáze 1 testy dokončeny!"

phase2-test: ## Test implementace Fáze 2 (M1 optimalizace)
	@echo "🧪 Testování Fáze 2 - M1 Optimalizace..."
	python -m pytest tests/phase2/ -v
	@echo "✅ Fáze 2 testy dokončeny!"
