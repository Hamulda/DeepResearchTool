# Research Agent - Production-Ready Makefile
# Automatizace běžných úloh pro development a CI/CD

.PHONY: help install install-dev lint format type-check test test-unit test-integration test-all clean security setup-env

# Default target
help:
	@echo "DeepResearchTool - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup-env        Setup environment from template"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting (ruff)"
	@echo "  format           Format code (black + ruff)"
	@echo "  type-check       Run type checking (mypy)"
	@echo "  security         Run security scans (bandit + safety)"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-fast        Run fast tests only (exclude slow)"
	@echo "  test-coverage    Run tests with detailed coverage"
	@echo ""
	@echo "Development:"
	@echo "  clean            Clean cache and temporary files"
	@echo "  docs             Generate documentation"
	@echo "  run-dev          Run development server"
	@echo ""

# ========================================
# SETUP & INSTALLATION
# ========================================

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-xdist
	pip install ruff black mypy isort
	pip install bandit safety
	pip install pre-commit

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		echo "Created .env from template. Please fill in your actual values."; \
	else \
		echo ".env already exists. Skipping."; \
	fi

# ========================================
# CODE QUALITY
# ========================================

lint:
	@echo "Running ruff linter..."
	ruff check src/ tests/
	@echo "Running import sorting check..."
	isort --check-only --diff src/ tests/

format:
	@echo "Formatting code with black..."
	black src/ tests/
	@echo "Formatting imports with isort..."
	isort src/ tests/
	@echo "Auto-fixing with ruff..."
	ruff check --fix src/ tests/

type-check:
	@echo "Running mypy type checking..."
	mypy src/ --config-file mypy.ini

security:
	@echo "Running bandit security scan..."
	bandit -r src/ -f txt
	@echo "Checking for known vulnerabilities..."
	safety check

# ========================================
# TESTING
# ========================================

test: test-unit test-integration
	@echo "All tests completed!"

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short

test-integration:
	@echo "Running integration tests..."
	pytest tests/test_integration_complete.py -v --tb=short -m "not slow and not external"

test-fast:
	@echo "Running fast tests only..."
	pytest tests/ -v --tb=short -m "not slow and not external and not ai_dependent"

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ \
		--cov=src \
		--cov-report=html \
		--cov-report=term \
		--cov-report=xml \
		--cov-branch \
		--cov-fail-under=80 \
		-v

test-all:
	@echo "Running complete test suite..."
	pytest tests/ -v --tb=short --maxfail=5

# ========================================
# DEVELOPMENT
# ========================================

clean:
	@echo "Cleaning cache and temporary files..."
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
	rm -f coverage.xml
	@echo "Cleanup completed!"

docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here
	@echo "Documentation generation not yet implemented"

run-dev:
	@echo "Starting development server..."
	python main.py --debug

# ========================================
# CI/CD HELPERS
# ========================================

ci-setup: install-dev setup-env
	@echo "CI environment setup completed"

ci-lint: lint type-check
	@echo "CI linting completed"

ci-test: test-fast
	@echo "CI testing completed"

ci-security: security
	@echo "CI security scanning completed"

ci-all: ci-setup ci-lint ci-security ci-test
	@echo "Complete CI pipeline completed successfully!"

# ========================================
# DOCKER
# ========================================

docker-build:
	@echo "Building Docker image..."
	docker build -t deepresearchtool:latest .

docker-build-prod:
	@echo "Building production Docker image..."
	docker build -f Dockerfile.production -t deepresearchtool:production .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8080:8080 --env-file .env deepresearchtool:latest

docker-compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up -d

docker-compose-down:
	@echo "Stopping services..."
	docker-compose down

# ========================================
# DATABASE
# ========================================

db-migrate:
	@echo "Running database migrations..."
	# Add migration commands here
	@echo "Database migrations not yet implemented"

db-seed:
	@echo "Seeding database with test data..."
	# Add seed commands here
	@echo "Database seeding not yet implemented"

# ========================================
# BENCHMARKS
# ========================================

benchmark:
	@echo "Running performance benchmarks..."
	pytest tests/ -v -m "performance" --benchmark-only

benchmark-compare:
	@echo "Comparing benchmark results..."
	pytest-benchmark compare

# ========================================
# RELEASE
# ========================================

version-bump-patch:
	@echo "Bumping patch version..."
	# Add version bumping logic here
	@echo "Version bumping not yet implemented"

version-bump-minor:
	@echo "Bumping minor version..."
	# Add version bumping logic here
	@echo "Version bumping not yet implemented"

release-prep: ci-all
	@echo "Preparing release..."
	@echo "Release preparation completed!"

# Research Agent - Production-Ready Makefile
.PHONY: help install test lint format docker-build docker-up validate demo clean

# Default help target
help: ## Show this help message
	@echo "Research Agent - Production Commands"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install all dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest-cov ruff mypy pre-commit bandit

# Code quality
lint: ## Run linting checks
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format: ## Format code with ruff
	ruff format src/ tests/

security-scan: ## Run security scan
	bandit -r src/ -f json -o bandit-report.json
	@echo "Security report saved to bandit-report.json"

# Testing
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/test_integration_complete.py -v

test-evaluation: ## Run evaluation pipeline tests
	pytest tests/test_evaluation_pipeline.py -v

test-coverage: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=xml
	@echo "Coverage report saved to htmlcov/index.html"

# Golden Dataset and Evaluation
evaluation-quick: ## Run quick evaluation on sample dataset
	python -m pytest tests/test_evaluation_pipeline.py::TestResearchAgentEvaluation::test_single_query_evaluation -v

evaluation-full: ## Run full evaluation on Golden Dataset
	CI=true python -m pytest tests/test_evaluation_pipeline.py::TestCIIntegration::test_ci_evaluation_with_thresholds -v

golden-dataset-validate: ## Validate Golden Dataset structure
	python -c "import json; data=json.load(open('evaluation/golden_dataset.json')); print(f'✅ Golden Dataset: {len(data)} questions')"

# Docker operations
docker-build: ## Build production Docker image
	docker build -f Dockerfile.production -t research-agent:latest .

docker-build-dev: ## Build development Docker image
	docker build -t research-agent:dev .

docker-up: ## Start full observability stack
	docker-compose -f docker-compose.observability.yml up -d

docker-down: ## Stop observability stack
	docker-compose -f docker-compose.observability.yml down

docker-logs: ## Show Langfuse logs
	docker-compose -f docker-compose.observability.yml logs -f langfuse

langfuse-setup: ## Initialize Langfuse with demo data
	@echo "🚀 Starting Langfuse observability stack..."
	docker-compose -f docker-compose.observability.yml up -d
	@echo "⏳ Waiting for Langfuse to be ready..."
	@timeout 60 bash -c 'until curl -f http://localhost:3000/api/health 2>/dev/null; do sleep 2; done' || echo "Timeout waiting for Langfuse"
	@echo "✅ Langfuse ready at http://localhost:3000"

# Validation and demos
validate: ## Run production readiness validation
	python scripts/validate_production_readiness.py

demo: ## Run enhanced research agent demo
	python demo_enhanced_research_agent.py

demo-committee: ## Demo expert committee architecture
	python -c "import asyncio; from src.graph.expert_committee import ExpertCommitteeGraph; print('Expert Committee Demo - see demo_enhanced_research_agent.py')"

# Production deployment preparation
prod-check: validate test lint security-scan ## Complete production readiness check
	@echo "🎉 Production readiness check completed!"

k8s-deploy: ## Deploy to Kubernetes (requires kubectl)
	@echo "🚀 Deploying to Kubernetes..."
	kubectl apply -f k8s/
	kubectl rollout status deployment/research-agent

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods -l app=research-agent
	kubectl get services -l app=research-agent

# Database operations
pgvector-setup: ## Setup PGVector database
	@echo "🗄️ Setting up PGVector database..."
	@echo "This requires PostgreSQL with pgvector extension"
	@echo "Run: CREATE EXTENSION vector; in your PostgreSQL database"

chroma-backup: ## Backup ChromaDB data
	@if [ -d "./chroma_db" ]; then \
		tar -czf chroma_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz chroma_db/; \
		echo "✅ ChromaDB backed up"; \
	else \
		echo "❌ ChromaDB directory not found"; \
	fi

# Development helpers
dev-setup: install-dev langfuse-setup ## Complete development environment setup
	@echo "🎯 Development environment ready!"
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and fill in your API keys"
	@echo "2. Run 'make demo' to test the system"
	@echo "3. Run 'make validate' to check production readiness"

env-template: ## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env from template - please fill in your API keys"; \
	else \
		echo "⚠️ .env already exists"; \
	fi

benchmark: ## Run performance benchmarks
	python scripts/bench_m1_performance.py

# Monitoring and observability
monitoring-up: ## Start monitoring stack (Prometheus + Grafana)
	docker-compose -f monitoring/docker-compose.monitoring.yml up -d

monitoring-down: ## Stop monitoring stack
	docker-compose -f monitoring/docker-compose.monitoring.yml down

logs: ## Show application logs
	docker-compose -f docker-compose.observability.yml logs -f research-agent

metrics: ## Show current metrics
	@echo "📊 Application Metrics:"
	@echo "Langfuse Dashboard: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3001"

# Cleanup
clean: ## Clean up temporary files and containers
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	docker system prune -f

clean-data: ## Clean up all data (databases, caches)
	@echo "⚠️ This will delete all data including ChromaDB and caches!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	rm -rf chroma_db/ demo_chroma_db/ research_cache/
	docker-compose -f docker-compose.observability.yml down -v

# Release management
version: ## Show current version info
	@echo "Research Agent Production System"
	@echo "Version: $(shell git describe --tags --always --dirty 2>/dev/null || echo 'dev')"
	@echo "Commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
	@echo "Branch: $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"

release-notes: ## Generate release notes
	@echo "📝 Generating release notes..."
	@echo "# Release Notes" > RELEASE_NOTES.md
	@echo "" >> RELEASE_NOTES.md
	@echo "## Features Implemented:" >> RELEASE_NOTES.md
	@echo "- ✅ LLM Observability (Langfuse integration)" >> RELEASE_NOTES.md
	@echo "- ✅ Automated Evaluation Pipeline (Golden Dataset + RAG Triad)" >> RELEASE_NOTES.md
	@echo "- ✅ Multi-Agent Expert Committee Architecture" >> RELEASE_NOTES.md
	@echo "- ✅ Production Scaling Plan" >> RELEASE_NOTES.md
	@echo "- ✅ CI/CD Pipeline with Regression Detection" >> RELEASE_NOTES.md
	@echo "- ✅ Docker & Kubernetes Support" >> RELEASE_NOTES.md
	@echo "" >> RELEASE_NOTES.md
	@echo "✅ Release notes generated in RELEASE_NOTES.md"

# Quick start commands
quick-start: env-template dev-setup demo ## Complete quick start setup
	@echo "🎊 Quick start completed!"

production-deploy: prod-check docker-build ## Build and validate for production
	@echo "🚀 Ready for production deployment!"
	@echo "Next: Push to registry and deploy with k8s-deploy"
