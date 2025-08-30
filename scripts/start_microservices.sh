#!/bin/bash

# Spouštěcí skript pro mikroslužbovou architekturu DeepResearchTool
# Fáze 1: Základní architektura a klíčová infrastruktura

set -e

echo "🚀 Spouštění DeepResearchTool mikroslužeb..."

# Kontrola Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker není nainstalován. Prosím nainstalujte Docker."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose není nainstalován. Prosím nainstalujte Docker Compose."
    exit 1
fi

# Vytvoření potřebných adresářů
echo "📁 Vytváření adresářů..."
mkdir -p data
mkdir -p research_cache
mkdir -p logs

# Build a spuštění služeb
echo "🔨 Buildování Docker images..."
docker-compose -f docker-compose.microservices.yml build

echo "🏃 Spouštění služeb..."
docker-compose -f docker-compose.microservices.yml up -d

# Čekání na spuštění služeb
echo "⏳ Čekání na inicializaci služeb..."
sleep 10

# Kontrola zdraví služeb
echo "🔍 Kontrola zdraví služeb..."

# Test Redis
if docker-compose -f docker-compose.microservices.yml exec -T task-queue-broker redis-cli ping | grep -q PONG; then
    echo "✅ Redis broker je zdravý"
else
    echo "❌ Redis broker není dostupný"
fi

# Test API Gateway
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API Gateway je zdravý"
else
    echo "❌ API Gateway není dostupný"
fi

# Test Vector DB
if curl -s http://localhost:6333/health > /dev/null; then
    echo "✅ Vector DB je zdravý"
else
    echo "❌ Vector DB není dostupný"
fi

echo ""
echo "🎉 DeepResearchTool mikroslužby jsou spuštěny!"
echo ""
echo "📊 Dostupné služby:"
echo "   • API Gateway: http://localhost:8000"
echo "   • Vector DB:   http://localhost:6333"
echo "   • Redis:       localhost:6379"
echo ""
echo "📖 Použití:"
echo "   curl -X POST http://localhost:8000/scrape -H 'Content-Type: application/json' -d '{\"url\": \"https://example.com\"}'"
echo ""
echo "🔧 Správa:"
echo "   docker-compose -f docker-compose.microservices.yml logs -f    # Zobrazit logy"
echo "   docker-compose -f docker-compose.microservices.yml stop       # Zastavit služby"
echo "   docker-compose -f docker-compose.microservices.yml down       # Smazat kontejnery"
