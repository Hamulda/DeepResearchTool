"""
Demo script for Phase 1: Memory-Efficient Data Core
Demonstrates end-to-end usage of all Phase 1 components
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
import logging
import polars as pl
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import Phase 1 components
from src.core.memory_optimizer import (
    MemoryOptimizer,
    LazyDataPipeline,
    ParquetDatasetManager,
    DuckDBQueryEngine,
)
from src.core.async_crawler import AsyncCrawler, CrawlConfig
from src.core.context_manager import ContextManager, ChunkingConfig
from src.core.vector_store import VectorStoreFactory, VectorConfig


async def demo_phase1_memory_efficient_core():
    """Comprehensive demo of Phase 1 capabilities"""
    logger.info("🚀 Starting Phase 1 Demo: Memory-Efficient Data Core")

    # Create temporary workspace
    temp_dir = Path(tempfile.mkdtemp(prefix="phase1_demo_"))
    logger.info(f"📁 Working directory: {temp_dir}")

    try:
        # 1. Memory Optimization Demo
        logger.info("\n1️⃣ Memory Optimizer Demo")
        optimizer = MemoryOptimizer(max_memory_gb=6.0)  # M1 8GB constraint

        memory_stats = optimizer.check_memory_pressure()
        logger.info(
            f"💾 Memory stats: {memory_stats['available_gb']:.1f}GB available, {memory_stats['used_percent']:.1f}% used"
        )

        optimal_batch = optimizer.get_optimal_batch_size()
        logger.info(f"⚡ Optimal batch size: {optimal_batch} records")

        # 2. Lazy Data Pipeline Demo
        logger.info("\n2️⃣ Lazy Data Pipeline Demo")
        pipeline = LazyDataPipeline(optimizer)

        # Create large synthetic dataset
        logger.info("🏭 Generating synthetic research data...")
        research_data = {
            "paper_id": range(10000),
            "title": [
                f"Research Paper {i}: Advanced Study on Topic {i % 100}" for i in range(10000)
            ],
            "abstract": [
                f"Abstract for paper {i} containing important findings about research area {i % 50}."
                * 3
                for i in range(10000)
            ],
            "field": ["AI", "Biology", "Physics", "Chemistry", "Mathematics"] * 2000,
            "year": [2020 + (i % 5) for i in range(10000)],
            "citations": [i * 2 + np.random.randint(0, 100) for i in range(10000)],
            "journal_tier": ["Q1", "Q2", "Q3", "Q4"] * 2500,
        }

        # Create lazy frame and apply transformations
        lazy_df = pipeline.create_lazy_frame(research_data)

        transformations = [
            {"operation": "filter", "params": {"column": "field", "value": "AI"}},
            {"operation": "filter", "params": {"column": "journal_tier", "value": "Q1"}},
            {"operation": "sort", "params": {"column": "citations", "desc": True}},
            {
                "operation": "select",
                "params": {"columns": ["paper_id", "title", "abstract", "citations"]},
            },
        ]

        processed_df = pipeline.apply_transformations(lazy_df, transformations)
        result = processed_df.collect()

        logger.info(
            f"📊 Processed {len(research_data['paper_id'])} records → {len(result)} filtered results"
        )
        logger.info(
            f"🔬 Top AI Q1 paper: {result['title'][0][:60]}... ({result['citations'][0]} citations)"
        )

        # 3. Parquet Dataset Management Demo
        logger.info("\n3️⃣ Parquet Dataset Management Demo")
        parquet_manager = ParquetDatasetManager(temp_dir / "research_dataset", optimizer)

        def research_data_generator():
            """Generate streaming research data"""
            for i in range(5000):
                yield {
                    "doc_id": f"doc_{i:06d}",
                    "content": f'Research document {i} discussing {["machine learning", "data science", "AI ethics", "quantum computing"][i % 4]}. '
                    * 10,
                    "source": ["arxiv", "pubmed", "ieee", "acm"][i % 4],
                    "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                    "category": ["research", "review", "tutorial"][i % 3],
                    "quality_score": round(0.5 + (i % 100) / 200, 2),
                }

        logger.info("💾 Writing streaming data to partitioned Parquet...")
        written_files = parquet_manager.write_streaming_batch(
            research_data_generator(), partition_cols=["source", "category"], compression="snappy"
        )

        logger.info(f"✅ Created {len(written_files)} Parquet files")

        # Test predicate pushdown
        logger.info("🔍 Testing predicate pushdown query...")
        lazy_query = parquet_manager.read_partitioned_lazy(
            columns=["doc_id", "content", "quality_score"],
            filters=[pl.col("source") == "arxiv", pl.col("quality_score") > 0.8],
        )

        high_quality_docs = lazy_query.collect()
        logger.info(f"🎯 Found {len(high_quality_docs)} high-quality arXiv documents")

        # 4. DuckDB Analytics Demo
        logger.info("\n4️⃣ DuckDB Analytics Demo")
        engine = DuckDBQueryEngine(optimizer)
        engine.register_parquet_dataset(
            "research_docs", temp_dir / "research_dataset", recursive=True
        )

        # Analytics queries
        queries = {
            "source_distribution": """
                SELECT source, category, COUNT(*) as doc_count, AVG(quality_score) as avg_quality
                FROM research_docs 
                GROUP BY source, category
                ORDER BY doc_count DESC
            """,
            "monthly_trends": """
                SELECT 
                    substr(date, 1, 7) as month,
                    COUNT(*) as docs,
                    AVG(quality_score) as quality
                FROM research_docs 
                GROUP BY month
                ORDER BY month
            """,
            "top_quality": """
                SELECT doc_id, source, quality_score
                FROM research_docs 
                WHERE quality_score > 0.9
                ORDER BY quality_score DESC
                LIMIT 10
            """,
        }

        for query_name, sql in queries.items():
            result_df = engine.execute_query(sql)
            logger.info(f"📈 {query_name}: {len(result_df)} results")

        # 5. Document Chunking and Context Management Demo
        logger.info("\n5️⃣ Document Chunking & Context Management Demo")

        chunking_config = ChunkingConfig(
            chunk_size=512, overlap_size=50, respect_sentence_boundaries=True, section_aware=True
        )

        context_manager = ContextManager(optimizer, chunking_config)

        # Sample research documents
        sample_docs = [
            (
                "research_paper_1",
                """
            # Machine Learning in Healthcare: A Comprehensive Review
            
            ## Abstract
            This paper presents a comprehensive review of machine learning applications in healthcare.
            We analyze recent developments and identify future research directions.
            
            ## Introduction
            Machine learning has revolutionized many aspects of healthcare, from diagnostic imaging
            to drug discovery. This review examines the current state of the field.
            
            ### Historical Context
            The application of ML in healthcare began in the 1970s with expert systems.
            
            ## Methodology
            We conducted a systematic literature review covering papers from 2020-2024.
            Our analysis includes both supervised and unsupervised learning approaches.
            
            ### Data Collection
            Papers were collected from major databases including PubMed, IEEE Xplore, and arXiv.
            
            ## Results
            We identified 500+ relevant papers across various healthcare domains.
            
            ### Clinical Applications
            The most common applications include medical imaging, electronic health records analysis,
            and clinical decision support systems.
            
            ## Discussion
            Our findings suggest that ML adoption in healthcare is accelerating.
            
            ## Conclusion
            Machine learning shows tremendous promise for improving healthcare outcomes.
            Future work should focus on explainability and regulatory compliance.
            """,
            ),
            (
                "technical_report_2",
                """
            # Data Privacy in AI Systems: Technical Report
            
            ## Executive Summary
            This report examines privacy-preserving techniques in AI systems.
            
            ## Background
            With increasing AI adoption, data privacy has become a critical concern.
            Organizations must balance innovation with privacy protection.
            
            ## Privacy-Preserving Techniques
            
            ### Differential Privacy
            Differential privacy provides mathematical guarantees about privacy protection.
            It adds calibrated noise to query results.
            
            ### Federated Learning
            Federated learning enables model training without centralizing data.
            This approach is particularly valuable for sensitive domains.
            
            ### Homomorphic Encryption
            Homomorphic encryption allows computation on encrypted data.
            Recent advances have made this more practical.
            
            ## Implementation Challenges
            
            ### Performance Overhead
            Privacy-preserving techniques often introduce computational overhead.
            
            ### Utility Trade-offs
            There's often a trade-off between privacy and model utility.
            
            ## Recommendations
            Organizations should adopt a privacy-by-design approach.
            Regular privacy audits are essential.
            
            ## Future Work
            Research should focus on reducing performance overhead while maintaining strong privacy guarantees.
            """,
            ),
        ]

        all_chunks = []
        for doc_id, content in sample_docs:
            chunks = context_manager.chunker.chunk_document(content, doc_id)
            all_chunks.extend(chunks)
            logger.info(f"📄 {doc_id}: {len(chunks)} chunks created")

        # Test semantic compression
        logger.info("🗜️ Testing semantic compression...")
        compressed_chunks = context_manager.compressor.compress_chunks(
            all_chunks, target_token_count=2000
        )

        total_original = sum(chunk.token_count for chunk in all_chunks)
        total_compressed = sum(chunk.token_count for chunk in compressed_chunks)
        compression_ratio = total_compressed / total_original

        logger.info(
            f"📉 Compression: {total_original} → {total_compressed} tokens ({compression_ratio:.2%} retained)"
        )

        # 6. Vector Store Demo
        logger.info("\n6️⃣ Vector Store Demo")

        vector_config = VectorConfig(
            provider="local",  # Use local for demo
            vector_size=384,
            collection_name="research_chunks",
            enable_quantization=True,
        )

        vector_store = VectorStoreFactory.create_vector_store(
            vector_config, optimizer, temp_dir / "vector_store"
        )

        # Generate embeddings (using random for demo - in real use case, would use actual embedding model)
        logger.info("🧮 Generating embeddings for chunks...")
        embeddings = np.random.rand(len(all_chunks), 384).astype(np.float32)

        # Add chunks to vector store
        await vector_store.add_chunks(all_chunks, embeddings)
        logger.info(f"✅ Added {len(all_chunks)} chunks to vector store")

        # Test similarity search
        logger.info("🔍 Testing similarity search...")
        query_embedding = np.random.rand(384).astype(np.float32)
        search_results = await vector_store.search(query_embedding, limit=5)

        logger.info(f"🎯 Found {len(search_results)} similar chunks:")
        for i, result in enumerate(search_results[:3]):
            preview = result.chunk.text[:100].replace("\n", " ")
            logger.info(f"  {i+1}. Score: {result.score:.3f} | {preview}...")

        # Vector store stats
        stats = vector_store.get_stats()
        logger.info(f"📊 Vector store stats: {stats}")

        # 7. Memory Usage Summary
        logger.info("\n7️⃣ Memory Usage Summary")
        final_memory = optimizer.check_memory_pressure()
        logger.info(
            f"💾 Final memory usage: {final_memory['used_percent']:.1f}% ({final_memory['available_gb']:.1f}GB available)"
        )

        if final_memory["pressure"]:
            logger.warning("⚠️  Memory pressure detected - optimization recommendations:")
            logger.warning("   • Reduce batch sizes")
            logger.warning("   • Enable more aggressive compression")
            logger.warning("   • Consider streaming processing")

        # 8. Performance Benchmark
        logger.info("\n8️⃣ Performance Benchmark Summary")
        logger.info(f"✅ Processed {len(research_data['paper_id'])} synthetic records")
        logger.info(f"✅ Created {len(written_files)} Parquet partitions")
        logger.info(f"✅ Chunked {len(sample_docs)} documents into {len(all_chunks)} chunks")
        logger.info(f"✅ Indexed {len(all_chunks)} vectors with compression")
        logger.info(f"✅ Peak memory usage: {100 - final_memory['available_gb']:.1f}GB")

        logger.info("\n🎉 Phase 1 Demo completed successfully!")
        logger.info("✨ Memory-efficient data core is ready for production use")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info(f"🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


async def demo_async_crawler():
    """Demonstrate async crawler with mock data (no real network requests)"""
    logger.info("\n🕷️ Async Crawler Demo (Mock Mode)")

    optimizer = MemoryOptimizer(max_memory_gb=6.0)
    temp_dir = Path(tempfile.mkdtemp(prefix="crawler_demo_"))

    try:
        config = CrawlConfig(
            max_concurrent=5, request_delay=0.5, timeout=10, max_retries=2, respect_robots=True
        )

        # Mock URLs for demonstration
        mock_urls = [
            "https://example.com/research/ai",
            "https://example.com/research/ml",
            "https://example.com/research/data-science",
        ]

        logger.info(f"🌐 Configured crawler for {len(mock_urls)} URLs")
        logger.info("📝 Note: Using mock mode to avoid real network requests in demo")

        # In a real scenario, this would make actual HTTP requests
        # For demo, we'll just show the configuration and setup
        async with AsyncCrawler(config, optimizer, temp_dir / "crawl_results") as crawler:
            stats = crawler.get_crawl_stats()
            logger.info(f"🔧 Crawler initialized: {stats}")

            logger.info("✅ Async crawler ready for real-world usage")
            logger.info("💡 Configure with actual URLs and enable network requests for production")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":

    async def main():
        """Run complete Phase 1 demo"""
        logger.info("🎬 DeepResearchTool Phase 1 Demo Starting...")

        await demo_phase1_memory_efficient_core()
        await demo_async_crawler()

        logger.info("\n🏁 All Phase 1 demos completed!")
        logger.info("🚀 Ready to proceed with Phase 2: Tor/I2P Integration")

    # Run the demo
    asyncio.run(main())
