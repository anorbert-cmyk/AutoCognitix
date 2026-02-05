#!/usr/bin/env python3
"""
Performance Benchmark Script for AutoCognitix.

Tests and measures performance across the platform:
- API response times
- Database query performance
- Vector search latency
- Embedding generation speed
- Cache hit/miss rates

Usage:
    python scripts/benchmark.py --all              # Run all benchmarks
    python scripts/benchmark.py --api              # API benchmarks only
    python scripts/benchmark.py --db               # Database benchmarks only
    python scripts/benchmark.py --vector           # Vector search benchmarks
    python scripts/benchmark.py --embedding        # Embedding benchmarks
    python scripts/benchmark.py --report report.json  # Save report to file
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@dataclass
class BenchmarkResult:
    """Single benchmark test result."""

    name: str
    category: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    success_rate: float
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    timestamp: str
    duration_seconds: float
    environment: str
    results: List[BenchmarkResult]
    summary: Dict[str, Any]


class PerformanceBenchmark:
    """
    Performance benchmark runner for AutoCognitix.

    Provides comprehensive testing of:
    - API endpoints (response times, throughput)
    - Database queries (PostgreSQL, Neo4j)
    - Vector search (Qdrant)
    - Embedding generation (huBERT)
    - Cache performance (Redis)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        iterations: int = 100,
        warmup_iterations: int = 5,
    ) -> None:
        """
        Initialize benchmark runner.

        Args:
            base_url: API base URL
            iterations: Number of iterations per test
            warmup_iterations: Warmup iterations before measurement
        """
        self.base_url = base_url
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.results: List[BenchmarkResult] = []
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    def _calculate_percentile(self, times: List[float], percentile: float) -> float:
        """Calculate percentile from sorted times list."""
        if not times:
            return 0.0
        sorted_times = sorted(times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]

    async def _benchmark_async(
        self,
        name: str,
        category: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run async benchmark with statistics.

        Args:
            name: Benchmark name
            category: Benchmark category
            func: Async function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            BenchmarkResult with timing statistics
        """
        times: List[float] = []
        errors: List[str] = []

        # Warmup iterations (not measured)
        for _ in range(self.warmup_iterations):
            try:
                await func(*args, **kwargs)
            except Exception:
                pass

        # Measured iterations
        for i in range(self.iterations):
            start = time.perf_counter()
            try:
                await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                times.append(elapsed)
            except Exception as e:
                errors.append(f"Iteration {i}: {str(e)}")

        # Calculate statistics
        if times:
            total_time = sum(times)
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            p50 = self._calculate_percentile(times, 50)
            p95 = self._calculate_percentile(times, 95)
            p99 = self._calculate_percentile(times, 99)
            success_rate = len(times) / self.iterations * 100
        else:
            total_time = avg_time = min_time = max_time = std_dev = 0
            p50 = p95 = p99 = 0
            success_rate = 0

        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=self.iterations,
            total_time_ms=round(total_time, 2),
            avg_time_ms=round(avg_time, 2),
            min_time_ms=round(min_time, 2),
            max_time_ms=round(max_time, 2),
            std_dev_ms=round(std_dev, 2),
            p50_ms=round(p50, 2),
            p95_ms=round(p95, 2),
            p99_ms=round(p99, 2),
            success_rate=round(success_rate, 2),
            errors=errors[:5],  # Keep first 5 errors only
        )

        self.results.append(result)
        return result

    # =========================================================================
    # API Benchmarks
    # =========================================================================

    async def benchmark_health_check(self) -> BenchmarkResult:
        """Benchmark health check endpoint (baseline)."""

        async def health_check():
            response = await self.client.get("/health")
            response.raise_for_status()

        return await self._benchmark_async(
            "Health Check",
            "api",
            health_check,
        )

    async def benchmark_dtc_search(self) -> BenchmarkResult:
        """Benchmark DTC search endpoint."""

        async def dtc_search():
            response = await self.client.get(
                "/api/v1/dtc/search",
                params={"q": "P0101", "limit": 20},
            )
            response.raise_for_status()

        return await self._benchmark_async(
            "DTC Search",
            "api",
            dtc_search,
        )

    async def benchmark_dtc_search_semantic(self) -> BenchmarkResult:
        """Benchmark DTC semantic search (uses embeddings)."""

        async def dtc_search_semantic():
            response = await self.client.get(
                "/api/v1/dtc/search",
                params={
                    "q": "motor nehezen indul hidegben",
                    "use_semantic": True,
                    "limit": 10,
                },
            )
            response.raise_for_status()

        return await self._benchmark_async(
            "DTC Semantic Search",
            "api",
            dtc_search_semantic,
        )

    async def benchmark_dtc_detail(self) -> BenchmarkResult:
        """Benchmark DTC detail endpoint."""

        async def dtc_detail():
            response = await self.client.get("/api/v1/dtc/P0101")
            response.raise_for_status()

        return await self._benchmark_async(
            "DTC Detail",
            "api",
            dtc_detail,
        )

    async def benchmark_dtc_detail_cached(self) -> BenchmarkResult:
        """Benchmark DTC detail with cache (second request)."""
        # Prime the cache
        await self.client.get("/api/v1/dtc/P0101")

        async def dtc_detail_cached():
            response = await self.client.get("/api/v1/dtc/P0101")
            response.raise_for_status()

        return await self._benchmark_async(
            "DTC Detail (Cached)",
            "api",
            dtc_detail_cached,
        )

    async def benchmark_vehicle_makes(self) -> BenchmarkResult:
        """Benchmark vehicle makes endpoint."""

        async def vehicle_makes():
            response = await self.client.get("/api/v1/vehicles/makes")
            response.raise_for_status()

        return await self._benchmark_async(
            "Vehicle Makes List",
            "api",
            vehicle_makes,
        )

    async def benchmark_quick_analyze(self) -> BenchmarkResult:
        """Benchmark quick analyze endpoint."""

        async def quick_analyze():
            response = await self.client.post(
                "/api/v1/diagnosis/quick-analyze",
                params={"dtc_codes": ["P0101", "P0171"]},
            )
            response.raise_for_status()

        return await self._benchmark_async(
            "Quick Analyze",
            "api",
            quick_analyze,
        )

    async def benchmark_categories(self) -> BenchmarkResult:
        """Benchmark DTC categories endpoint (static data)."""

        async def categories():
            response = await self.client.get("/api/v1/dtc/categories/list")
            response.raise_for_status()

        return await self._benchmark_async(
            "DTC Categories",
            "api",
            categories,
        )

    # =========================================================================
    # Database Benchmarks
    # =========================================================================

    async def benchmark_database_connection(self) -> BenchmarkResult:
        """Benchmark database connection pool performance."""
        try:
            from app.db.postgres.session import get_db

            async def db_connection():
                async for db in get_db():
                    from sqlalchemy import text

                    result = await db.execute(text("SELECT 1"))
                    _ = result.scalar()

            return await self._benchmark_async(
                "Database Connection",
                "database",
                db_connection,
            )
        except ImportError:
            return BenchmarkResult(
                name="Database Connection",
                category="database",
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                success_rate=0,
                errors=["Cannot import database modules"],
            )

    async def benchmark_dtc_query(self) -> BenchmarkResult:
        """Benchmark DTC code query from PostgreSQL."""
        try:
            from app.db.postgres.session import async_session_maker
            from app.db.postgres.repositories import DTCCodeRepository

            async def dtc_query():
                async with async_session_maker() as session:
                    repo = DTCCodeRepository(session)
                    await repo.get_by_code("P0101")

            return await self._benchmark_async(
                "DTC Query (PostgreSQL)",
                "database",
                dtc_query,
            )
        except ImportError:
            return BenchmarkResult(
                name="DTC Query (PostgreSQL)",
                category="database",
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                success_rate=0,
                errors=["Cannot import database modules"],
            )

    async def benchmark_dtc_search_query(self) -> BenchmarkResult:
        """Benchmark DTC search query with ILIKE."""
        try:
            from app.db.postgres.session import async_session_maker
            from app.db.postgres.repositories import DTCCodeRepository

            async def dtc_search_query():
                async with async_session_maker() as session:
                    repo = DTCCodeRepository(session)
                    await repo.search("motor", limit=20)

            return await self._benchmark_async(
                "DTC Search Query",
                "database",
                dtc_search_query,
            )
        except ImportError:
            return BenchmarkResult(
                name="DTC Search Query",
                category="database",
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                success_rate=0,
                errors=["Cannot import database modules"],
            )

    # =========================================================================
    # Vector Search Benchmarks
    # =========================================================================

    async def benchmark_vector_search(self) -> BenchmarkResult:
        """Benchmark Qdrant vector search."""
        try:
            from app.db.qdrant_client import qdrant_client
            from app.services.embedding_service import get_embedding_service

            # Generate a test embedding
            embedding_service = get_embedding_service()
            test_embedding = embedding_service.embed_text("motor hibakód teszt")

            async def vector_search():
                await qdrant_client.search_dtc(
                    query_vector=test_embedding,
                    limit=10,
                )

            return await self._benchmark_async(
                "Vector Search (Qdrant)",
                "vector",
                vector_search,
            )
        except ImportError:
            return BenchmarkResult(
                name="Vector Search (Qdrant)",
                category="vector",
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                success_rate=0,
                errors=["Cannot import Qdrant modules"],
            )

    # =========================================================================
    # Embedding Benchmarks
    # =========================================================================

    async def benchmark_embedding_single(self) -> BenchmarkResult:
        """Benchmark single text embedding generation."""
        try:
            from app.services.embedding_service import get_embedding_service

            embedding_service = get_embedding_service()
            # Warm up the model
            embedding_service.warmup()

            async def embed_single():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    embedding_service.embed_text,
                    "A motor nehezen indul hidegben és egyenetlenül jár alapjáraton.",
                )

            return await self._benchmark_async(
                "Embedding Single Text",
                "embedding",
                embed_single,
            )
        except ImportError:
            return BenchmarkResult(
                name="Embedding Single Text",
                category="embedding",
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                success_rate=0,
                errors=["Cannot import embedding modules"],
            )

    async def benchmark_embedding_batch(self) -> BenchmarkResult:
        """Benchmark batch embedding generation."""
        try:
            from app.services.embedding_service import get_embedding_service

            embedding_service = get_embedding_service()
            test_texts = [
                "Motor teljesítmény csökkenés",
                "Olajnyomás alacsony",
                "Hűtővíz hőmérséklet magas",
                "Fék kopásjelző aktív",
                "ABS lámpa világít",
            ]

            async def embed_batch():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    embedding_service.embed_batch,
                    test_texts,
                )

            return await self._benchmark_async(
                "Embedding Batch (5 texts)",
                "embedding",
                embed_batch,
            )
        except ImportError:
            return BenchmarkResult(
                name="Embedding Batch (5 texts)",
                category="embedding",
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                success_rate=0,
                errors=["Cannot import embedding modules"],
            )

    # =========================================================================
    # Cache Benchmarks
    # =========================================================================

    async def benchmark_cache_get(self) -> BenchmarkResult:
        """Benchmark Redis cache GET operation."""
        try:
            from app.db.redis_cache import get_cache_service

            cache = await get_cache_service()
            # Set a test value
            await cache.set("benchmark:test", {"data": "test"}, ttl=3600)

            async def cache_get():
                await cache.get("benchmark:test")

            result = await self._benchmark_async(
                "Cache GET",
                "cache",
                cache_get,
            )

            # Cleanup
            await cache.delete("benchmark:test")
            return result

        except ImportError:
            return BenchmarkResult(
                name="Cache GET",
                category="cache",
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                success_rate=0,
                errors=["Cannot import cache modules"],
            )

    async def benchmark_cache_set(self) -> BenchmarkResult:
        """Benchmark Redis cache SET operation."""
        try:
            from app.db.redis_cache import get_cache_service

            cache = await get_cache_service()
            test_data = {"code": "P0101", "description": "Test data"}

            async def cache_set():
                await cache.set(f"benchmark:test:{time.time_ns()}", test_data, ttl=60)

            result = await self._benchmark_async(
                "Cache SET",
                "cache",
                cache_set,
            )

            # Cleanup
            await cache.delete_pattern("benchmark:test:*")
            return result

        except ImportError:
            return BenchmarkResult(
                name="Cache SET",
                category="cache",
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                success_rate=0,
                errors=["Cannot import cache modules"],
            )

    # =========================================================================
    # Run All Benchmarks
    # =========================================================================

    async def run_api_benchmarks(self) -> List[BenchmarkResult]:
        """Run all API benchmarks."""
        print("\n=== API Benchmarks ===")
        results = []

        benchmarks = [
            ("Health Check", self.benchmark_health_check),
            ("DTC Search", self.benchmark_dtc_search),
            ("DTC Semantic Search", self.benchmark_dtc_search_semantic),
            ("DTC Detail", self.benchmark_dtc_detail),
            ("DTC Detail (Cached)", self.benchmark_dtc_detail_cached),
            ("Vehicle Makes", self.benchmark_vehicle_makes),
            ("Quick Analyze", self.benchmark_quick_analyze),
            ("DTC Categories", self.benchmark_categories),
        ]

        for name, func in benchmarks:
            print(f"Running: {name}...", end=" ", flush=True)
            try:
                result = await func()
                print(f"avg={result.avg_time_ms:.2f}ms, p95={result.p95_ms:.2f}ms")
                results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")

        return results

    async def run_database_benchmarks(self) -> List[BenchmarkResult]:
        """Run all database benchmarks."""
        print("\n=== Database Benchmarks ===")
        results = []

        benchmarks = [
            ("Database Connection", self.benchmark_database_connection),
            ("DTC Query", self.benchmark_dtc_query),
            ("DTC Search Query", self.benchmark_dtc_search_query),
        ]

        for name, func in benchmarks:
            print(f"Running: {name}...", end=" ", flush=True)
            try:
                result = await func()
                print(f"avg={result.avg_time_ms:.2f}ms, p95={result.p95_ms:.2f}ms")
                results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")

        return results

    async def run_vector_benchmarks(self) -> List[BenchmarkResult]:
        """Run all vector search benchmarks."""
        print("\n=== Vector Search Benchmarks ===")
        results = []

        print("Running: Vector Search...", end=" ", flush=True)
        try:
            result = await self.benchmark_vector_search()
            print(f"avg={result.avg_time_ms:.2f}ms, p95={result.p95_ms:.2f}ms")
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")

        return results

    async def run_embedding_benchmarks(self) -> List[BenchmarkResult]:
        """Run all embedding benchmarks."""
        print("\n=== Embedding Benchmarks ===")
        results = []

        benchmarks = [
            ("Single Embedding", self.benchmark_embedding_single),
            ("Batch Embedding", self.benchmark_embedding_batch),
        ]

        for name, func in benchmarks:
            print(f"Running: {name}...", end=" ", flush=True)
            try:
                result = await func()
                print(f"avg={result.avg_time_ms:.2f}ms, p95={result.p95_ms:.2f}ms")
                results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")

        return results

    async def run_cache_benchmarks(self) -> List[BenchmarkResult]:
        """Run all cache benchmarks."""
        print("\n=== Cache Benchmarks ===")
        results = []

        benchmarks = [
            ("Cache GET", self.benchmark_cache_get),
            ("Cache SET", self.benchmark_cache_set),
        ]

        for name, func in benchmarks:
            print(f"Running: {name}...", end=" ", flush=True)
            try:
                result = await func()
                print(f"avg={result.avg_time_ms:.2f}ms, p95={result.p95_ms:.2f}ms")
                results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")

        return results

    async def run_all(self) -> BenchmarkReport:
        """Run all benchmarks and generate report."""
        start_time = time.time()

        await self.run_api_benchmarks()
        await self.run_database_benchmarks()
        await self.run_vector_benchmarks()
        await self.run_embedding_benchmarks()
        await self.run_cache_benchmarks()

        duration = time.time() - start_time

        # Generate summary
        summary = self._generate_summary()

        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=round(duration, 2),
            environment="development",
            results=self.results,
            summary=summary,
        )

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary statistics."""
        if not self.results:
            return {}

        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        # Calculate category averages
        category_summary = {}
        for cat, results in categories.items():
            avg_times = [r.avg_time_ms for r in results if r.success_rate > 0]
            category_summary[cat] = {
                "tests": len(results),
                "avg_response_ms": round(statistics.mean(avg_times), 2) if avg_times else 0,
                "max_p95_ms": max((r.p95_ms for r in results), default=0),
            }

        # Overall statistics
        all_success_rates = [r.success_rate for r in self.results]
        all_p95 = [r.p95_ms for r in self.results if r.success_rate > 0]

        return {
            "total_tests": len(self.results),
            "overall_success_rate": round(statistics.mean(all_success_rates), 2),
            "max_p95_ms": max(all_p95, default=0),
            "categories": category_summary,
            "slowest_tests": [
                {"name": r.name, "avg_ms": r.avg_time_ms}
                for r in sorted(self.results, key=lambda x: x.avg_time_ms, reverse=True)[:5]
            ],
        }

    def print_report(self, report: BenchmarkReport) -> None:
        """Print benchmark report to console."""
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Duration: {report.duration_seconds:.2f} seconds")
        print(f"Environment: {report.environment}")
        print()

        # Results table
        print(f"{'Test Name':<35} {'Avg(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Success':<10}")
        print("-" * 75)

        for result in report.results:
            print(
                f"{result.name:<35} "
                f"{result.avg_time_ms:<10.2f} "
                f"{result.p95_ms:<10.2f} "
                f"{result.p99_ms:<10.2f} "
                f"{result.success_rate:<10.1f}%"
            )

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total tests: {report.summary.get('total_tests', 0)}")
        print(f"Overall success rate: {report.summary.get('overall_success_rate', 0)}%")
        print(f"Max P95 latency: {report.summary.get('max_p95_ms', 0):.2f}ms")

        print("\nCategory Breakdown:")
        for cat, stats in report.summary.get("categories", {}).items():
            print(f"  {cat}: {stats['tests']} tests, avg={stats['avg_response_ms']}ms")

        print("\nSlowest Tests:")
        for test in report.summary.get("slowest_tests", [])[:5]:
            print(f"  - {test['name']}: {test['avg_ms']:.2f}ms")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AutoCognitix Performance Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per test")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--api", action="store_true", help="Run API benchmarks")
    parser.add_argument("--db", action="store_true", help="Run database benchmarks")
    parser.add_argument("--vector", action="store_true", help="Run vector benchmarks")
    parser.add_argument("--embedding", action="store_true", help="Run embedding benchmarks")
    parser.add_argument("--cache", action="store_true", help="Run cache benchmarks")
    parser.add_argument("--report", help="Save report to JSON file")

    args = parser.parse_args()

    # Default to all if no specific flag
    if not any([args.api, args.db, args.vector, args.embedding, args.cache]):
        args.all = True

    print(f"AutoCognitix Performance Benchmark")
    print(f"Base URL: {args.url}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")

    async with PerformanceBenchmark(
        base_url=args.url,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
    ) as benchmark:
        start_time = time.time()

        if args.all or args.api:
            await benchmark.run_api_benchmarks()

        if args.all or args.db:
            await benchmark.run_database_benchmarks()

        if args.all or args.vector:
            await benchmark.run_vector_benchmarks()

        if args.all or args.embedding:
            await benchmark.run_embedding_benchmarks()

        if args.all or args.cache:
            await benchmark.run_cache_benchmarks()

        # Generate report
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=round(time.time() - start_time, 2),
            environment="development",
            results=benchmark.results,
            summary=benchmark._generate_summary(),
        )

        # Print report
        benchmark.print_report(report)

        # Save report if requested
        if args.report:
            with open(args.report, "w") as f:
                json.dump(
                    {
                        "timestamp": report.timestamp,
                        "duration_seconds": report.duration_seconds,
                        "environment": report.environment,
                        "results": [asdict(r) for r in report.results],
                        "summary": report.summary,
                    },
                    f,
                    indent=2,
                )
            print(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
