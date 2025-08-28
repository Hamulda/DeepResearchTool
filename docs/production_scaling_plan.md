"""
Plán škálování Research Agent pro produkční nasazení
Technická specifikace pro přechod z prototypu na enterprise řešení
"""

# PLÁN ŠKÁLOVÁNÍ PRO PRODUKCI
## Research Agent - Enterprise Architecture

### 1. VEKTOROVÁ DATABÁZE - MIGRACE Z CHROMADB

#### Současný stav:
- ChromaDB: In-memory/lokální persistence
- Omezení: Single-node, limitovaná škálovatelnost
- Vhodné pro: Prototyp, development

#### Cílová architektura:

##### Volba A: PGVector (PostgreSQL Extension)
**Výhody:**
- Nativní SQL podpora
- ACID compliance
- Mature ecosystem
- Horizontální škálování přes sharding
- Backup/recovery řešení

**Implementační kroky:**
1. Migrace dat z ChromaDB do PostgreSQL s pgvector
2. Optimalizace indexů (HNSW, IVFFlat)
3. Partitioning podle časových období
4. Read replicas pro query load balancing
5. Connection pooling (PgBouncer)

**Konfigurace:**
```sql
-- Optimalizace pro embedding vyhledávání
CREATE INDEX ON embeddings USING hnsw (vector vector_cosine_ops);
CREATE INDEX ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 1000);

-- Partitioning podle data
CREATE TABLE embeddings_2024 PARTITION OF embeddings 
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

##### Volba B: Pinecone (Managed Service)
**Výhody:**
- Fully managed
- Auto-scaling
- Multi-region support
- Built-in filtering a metadata

**Implementační kroky:**
1. Pinecone account setup
2. Index creation s optimalizovanými parametry
3. Batch migration utility
4. API integration layer
5. Cost monitoring

##### Volba C: Weaviate (Self-hosted/Cloud)
**Výhody:**
- Multi-modal support
- GraphQL API
- Semantic search capabilities
- Kubernetes native

#### Migrace strategie:
```python
# Migration utility
class VectorDBMigrator:
    async def migrate_chroma_to_pgvector(self):
        # 1. Export z ChromaDB
        # 2. Transform embeddings
        # 3. Batch insert do PostgreSQL
        # 4. Verify data integrity
        # 5. Switch traffic
        pass
```

### 2. SCRAPING INFRASTRUKTURA - ENTERPRISE GRADE

#### Současný stav:
- Firecrawl: API limits, rate limiting
- Omezení: Reliability, scale, legal compliance

#### Cílová architektura:

##### Volba A: Bright Data (Enterprise Proxy Network)
**Capabilities:**
- 72M+ IP addresses
- Residential/datacenter proxies
- Built-in CAPTCHA solving
- Legal compliance framework
- 99.9% uptime SLA

**Implementace:**
```python
class BrightDataScraper:
    def __init__(self):
        self.proxy_manager = BrightDataProxyManager()
        self.scraping_browser = ScrapingBrowser()
    
    async def scrape_with_rotation(self, urls: List[str]):
        # Automatic IP rotation
        # Anti-detection measures
        # Parallel processing
        pass
```

##### Volba B: Apify Platform
**Capabilities:**
- Pre-built scrapers
- Cloud infrastructure
- Automated scaling
- Data extraction pipelines

##### Volba C: ScrapingBee/Scraperapi
**Capabilities:**
- API-first approach
- JavaScript rendering
- CAPTCHA handling
- Global proxy network

#### Hybrid approach:
```python
class EnterpriseScrapingOrchestrator:
    def __init__(self):
        self.providers = {
            "bright_data": BrightDataScraper(),
            "apify": ApifyScraper(),
            "scraperapi": ScraperAPIScraper()
        }
    
    async def intelligent_routing(self, request):
        # Route based on:
        # - Website complexity
        # - Legal requirements
        # - Cost optimization
        # - Success rates
        pass
```

### 3. INFRASTRUKTURNÍ ARCHITEKTURA

#### Container Orchestration - Kubernetes

```yaml
# Production Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-agent
spec:
  replicas: 5
  selector:
    matchLabels:
      app: research-agent
  template:
    spec:
      containers:
      - name: research-agent
        image: research-agent:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

#### Load Balancing a API Gateway

```yaml
# NGINX Ingress s rate limiting
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: research-agent-ingress
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  rules:
  - host: api.research-agent.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: research-agent-service
            port:
              number: 8000
```

#### Caching Layer - Redis Cluster

```python
class DistributedCacheManager:
    def __init__(self):
        self.redis_cluster = RedisCluster(
            startup_nodes=[
                {"host": "redis-1", "port": "7000"},
                {"host": "redis-2", "port": "7000"},
                {"host": "redis-3", "port": "7000"}
            ]
        )
    
    async def cache_research_result(self, query_hash: str, result: Dict):
        # TTL based on query type
        ttl = self._calculate_ttl(result)
        await self.redis_cluster.setex(
            f"research:{query_hash}", 
            ttl, 
            json.dumps(result)
        )
```

### 4. MONITORING A OBSERVABILITY

#### Metrics Stack
```yaml
# Prometheus + Grafana + AlertManager
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
```

#### Custom Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
research_requests_total = Counter('research_requests_total', 'Total research requests')
research_duration = Histogram('research_duration_seconds', 'Research duration')
active_research_sessions = Gauge('active_research_sessions', 'Active research sessions')
```

### 5. SECURITY A COMPLIANCE

#### API Security
```python
class SecurityMiddleware:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.api_key_validator = APIKeyValidator()
        self.input_sanitizer = InputSanitizer()
    
    async def validate_request(self, request):
        # Rate limiting
        # API key validation
        # Input sanitization
        # SQL injection prevention
        # XSS protection
        pass
```

#### Data Privacy
```python
class PIIDetector:
    def __init__(self):
        self.nlp_model = spacy.load("en_core_web_sm")
    
    def detect_and_redact_pii(self, text: str) -> str:
        # Email detection
        # Phone number detection
        # SSN detection
        # Credit card detection
        return redacted_text
```

### 6. DEPLOYMENT STRATEGIE

#### Blue-Green Deployment
```bash
#!/bin/bash
# Blue-Green deployment script

CURRENT_COLOR=$(kubectl get service research-agent -o jsonpath='{.spec.selector.version}')
NEW_COLOR=$([ "$CURRENT_COLOR" = "blue" ] && echo "green" || echo "blue")

echo "Current: $CURRENT_COLOR, Deploying: $NEW_COLOR"

# Deploy new version
kubectl apply -f k8s/deployment-$NEW_COLOR.yaml

# Wait for readiness
kubectl rollout status deployment/research-agent-$NEW_COLOR

# Switch traffic
kubectl patch service research-agent -p '{"spec":{"selector":{"version":"'$NEW_COLOR'"}}}'

# Cleanup old version after validation
sleep 300
kubectl delete deployment research-agent-$CURRENT_COLOR
```

#### Canary Deployment s Istio
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: research-agent-canary
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: research-agent
        subset: v2
  - route:
    - destination:
        host: research-agent
        subset: v1
      weight: 90
    - destination:
        host: research-agent
        subset: v2
      weight: 10
```

### 7. COST OPTIMIZATION

#### Resource Right-sizing
```python
class ResourceOptimizer:
    async def analyze_usage_patterns(self):
        # CPU/Memory utilization analysis
        # Request pattern analysis
        # Scaling recommendations
        pass
    
    async def implement_hpa(self):
        # Horizontal Pod Autoscaler
        # Custom metrics scaling
        # Predictive scaling
        pass
```

#### Multi-cloud Strategy
```yaml
# Terraform pro multi-cloud deployment
resource "aws_eks_cluster" "primary" {
  name = "research-agent-primary"
  region = "us-east-1"
}

resource "google_container_cluster" "secondary" {
  name = "research-agent-secondary"
  location = "us-central1"
}

resource "azurerm_kubernetes_cluster" "tertiary" {
  name = "research-agent-tertiary"
  location = "East US"
}
```

### 8. DISASTER RECOVERY

#### Backup Strategy
```python
class BackupManager:
    async def create_full_backup(self):
        # Database backup
        # Vector embeddings backup
        # Configuration backup
        # Encrypted storage
        pass
    
    async def restore_from_backup(self, backup_id: str):
        # Point-in-time recovery
        # Cross-region restore
        # Validation checks
        pass
```

#### Multi-region Failover
```yaml
# Global Load Balancer konfigurace
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: research-agent-ssl
spec:
  domains:
    - api.research-agent.com

---
apiVersion: networking.gke.io/v1beta1
kind: FrontendConfig
metadata:
  name: research-agent-frontend
spec:
  redirectToHttps:
    enabled: true
```

### 9. IMPLEMENTAČNÍ TIMELINE

#### Fáze 1 (Měsíc 1-2): Foundation
- [ ] Kubernetes cluster setup
- [ ] CI/CD pipeline implementace
- [ ] Basic monitoring setup
- [ ] Security hardening

#### Fáze 2 (Měsíc 2-3): Data Layer
- [ ] PGVector migrace
- [ ] Backup/recovery řešení
- [ ] Performance optimization
- [ ] Data partitioning

#### Fáze 3 (Měsíc 3-4): Scaling
- [ ] Enterprise scraping integration
- [ ] Multi-region deployment
- [ ] Auto-scaling implementation
- [ ] Load testing

#### Fáze 4 (Měsíc 4-5): Optimization
- [ ] Cost optimization
- [ ] Performance tuning
- [ ] Advanced monitoring
- [ ] Compliance validation

#### Fáze 5 (Měsíc 5-6): Production Ready
- [ ] Disaster recovery testing
- [ ] Security audit
- [ ] Documentation completion
- [ ] Team training

### 10. SUCCESS METRICS

#### Technical KPIs
- Uptime: 99.9%
- Response time: < 2s (95th percentile)
- Throughput: 1000+ requests/minute
- Error rate: < 0.1%

#### Business KPIs
- Research quality scores: > 0.8
- User satisfaction: > 4.5/5
- Cost per research query: < $0.10
- Time to insight: < 30s

#### Operational KPIs
- Deployment frequency: Weekly
- Recovery time: < 30 minutes
- Change failure rate: < 5%
- Lead time: < 2 days
