# Deep Research Tool v2.0 Enhancement Plan
## Adaptive Retrieval, Fusion & Observability

### 🎯 OBJECTIVES
Implementace pokročilých funkcí pro zvýšení hloubky výzkumu, kvality analýzy a auditovatelnosti při zachování lokálního provozu na MacBook Air M1.

---

## 📋 IMPLEMENTATION CHECKLIST

### A. RETRIEVAL & FUSION
- [ ] **A1**: Per-query SearchParams pro Qdrant (hnsw_ef/exact routing)
- [ ] **A2**: LTR implementace (LightGBM) s RRF fallback
- [ ] **A3**: Temporal-aware ranking s recency scoring
- [ ] **A4**: Source diversity & authority scoring
- [ ] **A5**: A/B testing framework pro fusion metody

### B. CONTEXT & COMPRESSION
- [ ] **B1**: Adaptive chunking (structure + semantic boundaries)
- [ ] **B2**: Gated reranking (bi-encoder → cross-encoder)
- [ ] **B3**: Semantic cache s embedding keys
- [ ] **B4**: Result cache s TTL per doména
- [ ] **B5**: Cache invalidation při změně konfigu

### C. VERIFICATION & PROVENANCE
- [ ] **C1**: Contradiction sets v odpovědi s confidence scoring
- [ ] **C2**: WARC/CDX provenance tracking
- [ ] **C3**: Timestamp + hash pro každou citaci
- [ ] **C4**: Forensic JSON-LD export claim-graphu
- [ ] **C5**: Domain risk scoring

### D. OBSERVABILITY
- [ ] **D1**: OpenTelemetry tracing celého DAG
- [ ] **D2**: Per-node metriky (latence, cache-hit, tokeny)
- [ ] **D3**: OTLP/Jaeger export
- [ ] **D4**: Performance dashboards
- [ ] **D5**: Cost tracking per claim

### E. EVALUATION & QUALITY GATES
- [ ] **E1**: Domain-specific eval suites
- [ ] **E2**: Rozšířené metriky (recall@k, nDCG, groundedness)
- [ ] **E3**: Contradiction catch rate
- [ ] **E4**: Calibration curves
- [ ] **E5**: CI gates s minimálními prahy

### F. SECURITY & OSINT
- [ ] **F1**: Tor/Ahmia whitelists/blacklists
- [ ] **F2**: Sandbox bez JS execution
- [ ] **F3**: PII/GDPR redaction
- [ ] **F4**: Common Crawl integration
- [ ] **F5**: Audit logging

### G. M1 PERFORMANCE
- [ ] **G1**: Profilovaný batch sizing
- [ ] **G2**: Q4_K_M vs Q5_K_M routing
- [ ] **G3**: Speculative decoding (optional)
- [ ] **G4**: Dynamic contradiction pass
- [ ] **G5**: Feature flags pro performance

### H. DOCUMENTATION
- [ ] **H1**: ARCHITECTURE.md update
- [ ] **H2**: EVAL.md s benchmark results
- [ ] **H3**: SECURITY.md policy
- [ ] **H4**: OPERATIONS.md troubleshooting
- [ ] **H5**: CONFIG.md reference

---

## 🚀 IMPLEMENTATION PHASES

### Phase 1: Core Retrieval & Fusion (A1-A5)
**Estimated**: 2-3 days
- Adaptive Qdrant parameters
- LTR model implementation
- Temporal ranking

### Phase 2: Context & Compression (B1-B5)
**Estimated**: 2 days
- Adaptive chunking
- Gated reranking
- Smart caching

### Phase 3: Verification & Provenance (C1-C5)
**Estimated**: 2 days
- Contradiction detection
- Provenance tracking
- Forensic export

### Phase 4: Observability (D1-D5)
**Estimated**: 1-2 days
- OpenTelemetry integration
- Metrics collection
- Performance monitoring

### Phase 5: Evaluation & Security (E1-E5, F1-F5)
**Estimated**: 2-3 days
- Domain eval suites
- Security features
- Quality gates

### Phase 6: Performance & Documentation (G1-G5, H1-H5)
**Estimated**: 1-2 days
- M1 optimizations
- Comprehensive documentation

---

## 🎯 ACCEPTANCE CRITERIA

### Retrieval Performance
- [ ] Recall@10 improvement ≥15% vs baseline RRF
- [ ] nDCG@10 improvement ≥10% across 3 domains
- [ ] Latency under 60s (quick) / 300s (thorough)

### Quality Metrics
- [ ] Groundedness score ≥0.85
- [ ] Citation precision ≥0.90
- [ ] Contradiction catch rate ≥0.75
- [ ] Calibration error <0.1

### Performance Requirements
- [ ] Cache hit rate ≥70%
- [ ] Memory usage <16GB
- [ ] Token efficiency ≥85%

### Security & Compliance
- [ ] PII redaction 100%
- [ ] Audit trail complete
- [ ] Legal filter compliance

---

## 📁 FILE STRUCTURE

```
src/
├── retrieval/
│   ├── adaptive_params.py      # A1: Per-query Qdrant params
│   ├── ltr_fusion.py          # A2: LightGBM LTR model
│   └── temporal_ranking.py    # A3: Recency scoring
├── compression/
│   ├── adaptive_chunking.py   # B1: Structure+semantic chunking
│   └── gated_reranking.py     # B2: Bi-encoder → cross-encoder
├── cache/
│   ├── semantic_cache.py      # B3: Embedding-based cache
│   └── result_cache.py        # B4: TTL result cache
├── provenance/
│   ├── contradiction_sets.py  # C1: Pro/contra detection
│   ├── warc_tracking.py       # C2: WARC/CDX provenance
│   └── forensic_export.py     # C4: JSON-LD export
├── observability/
│   ├── otel_tracing.py        # D1: OpenTelemetry spans
│   ├── metrics_collector.py   # D2: Performance metrics
│   └── cost_tracker.py        # D5: Cost per claim
├── evaluation/
│   ├── domain_suites/         # E1: Domain-specific tests
│   ├── calibration.py         # E4: Calibration curves
│   └── ci_gates.py           # E5: Quality gates
└── security/
    ├── osint_sandbox.py       # F2: Safe OSINT execution
    ├── pii_redaction.py       # F3: PII/GDPR compliance
    └── audit_logger.py        # F5: Security audit
```

---

## 🧪 TESTING STRATEGY

### Unit Tests
- [ ] LTR model accuracy
- [ ] Chunking boundary detection
- [ ] Cache hit/miss logic
- [ ] Provenance tracking
- [ ] Metrics collection

### Integration Tests
- [ ] End-to-end DAG flow
- [ ] Multi-source fusion
- [ ] Cache performance
- [ ] Security filtering

### Performance Tests
- [ ] Latency benchmarks
- [ ] Memory profiling
- [ ] Cache efficiency
- [ ] M1 optimization

### Golden Tests
- [ ] Deterministic outputs
- [ ] Regression detection
- [ ] Quality baselines

---

## 📊 SUCCESS METRICS

| Metric | Baseline | Target | Critical |
|--------|----------|---------|----------|
| Recall@10 | 0.65 | 0.75 | 0.70 |
| nDCG@10 | 0.72 | 0.80 | 0.75 |
| Groundedness | 0.78 | 0.85 | 0.80 |
| Latency (quick) | 45s | 40s | 60s |
| Cache Hit Rate | 45% | 70% | 60% |
| Memory Usage | 12GB | 10GB | 16GB |

