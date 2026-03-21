"""Unit tests for app.db.redis_cache module."""

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.redis_cache import (
    CachePrefix,
    CacheTTL,
    RedisCacheService,
    cached,
)


# ---------------------------------------------------------------------------
# Fixture: fresh RedisCacheService (bypass singleton)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton before/after every test."""
    RedisCacheService._instance = None
    RedisCacheService._pool = None
    RedisCacheService._initialized = False
    yield
    RedisCacheService._instance = None
    RedisCacheService._pool = None
    RedisCacheService._initialized = False


@pytest.fixture
def service() -> RedisCacheService:
    svc = RedisCacheService()
    svc._connected = True
    svc._circuit_open = False
    svc._failure_count = 0
    svc._client = AsyncMock()
    return svc


# ===========================================================================
# CacheTTL constants
# ===========================================================================


class TestCacheTTL:
    def test_dtc_code_ttl(self):
        assert CacheTTL.DTC_CODE == 3600

    def test_dtc_search_ttl(self):
        assert CacheTTL.DTC_SEARCH == 900

    def test_vehicle_data_ttl(self):
        assert CacheTTL.VEHICLE_DATA == 86400

    def test_api_response_ttl(self):
        assert CacheTTL.API_RESPONSE == 300

    def test_nhtsa_data_ttl(self):
        assert CacheTTL.NHTSA_DATA == 21600

    def test_embeddings_ttl(self):
        assert CacheTTL.EMBEDDINGS == 3600

    def test_session_ttl(self):
        assert CacheTTL.SESSION == 1800


# ===========================================================================
# CachePrefix constants
# ===========================================================================


class TestCachePrefix:
    def test_dtc_code_prefix(self):
        assert CachePrefix.DTC_CODE == "dtc:code:"

    def test_rate_limit_prefix(self):
        assert CachePrefix.RATE_LIMIT == "ratelimit:"

    def test_embedding_prefix(self):
        assert CachePrefix.EMBEDDING == "embed:"


# ===========================================================================
# Singleton behaviour
# ===========================================================================


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        a = RedisCacheService()
        b = RedisCacheService()
        assert a is b

    def test_initialized_only_once(self):
        svc = RedisCacheService()
        assert svc._initialized is True
        svc._failure_count = 42
        svc2 = RedisCacheService()
        assert svc2._failure_count == 42  # same object, not re-initialised


# ===========================================================================
# connect / disconnect / warmup
# ===========================================================================


class TestConnection:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        svc = RedisCacheService()
        with (
            patch("app.db.redis_cache.ConnectionPool") as MockPool,
            patch("app.db.redis_cache.redis.Redis") as MockRedis,
        ):
            pool_instance = MagicMock()
            MockPool.from_url.return_value = pool_instance
            client_instance = AsyncMock()
            MockRedis.return_value = client_instance

            await svc.connect()

            assert svc._connected is True
            assert svc._circuit_open is False
            client_instance.ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, service):
        service._client.ping = AsyncMock()
        await service.connect()  # should be a no-op
        service._client.ping.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connect_failure_raises(self):
        svc = RedisCacheService()
        with patch("app.db.redis_cache.ConnectionPool") as MockPool:
            MockPool.from_url.side_effect = ConnectionError("refused")
            with pytest.raises(ConnectionError):
                await svc.connect()
            assert svc._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, service):
        RedisCacheService._pool = AsyncMock()
        await service.disconnect()
        assert service._connected is False
        assert service._client is None

    @pytest.mark.asyncio
    async def test_warmup_success(self, service):
        service._connected = False
        with patch.object(service, "connect", new_callable=AsyncMock):
            result = await service.warmup()
            assert result is True

    @pytest.mark.asyncio
    async def test_warmup_failure(self):
        svc = RedisCacheService()
        with patch.object(svc, "connect", side_effect=Exception("fail")):
            result = await svc.warmup()
            assert result is False


# ===========================================================================
# Circuit Breaker
# ===========================================================================


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_check_circuit_closed(self, service):
        assert await service._check_circuit() is True

    @pytest.mark.asyncio
    async def test_check_circuit_open(self, service):
        service._circuit_open = True
        assert await service._check_circuit() is False

    @pytest.mark.asyncio
    async def test_record_failure_opens_circuit(self, service):
        service._max_failures = 3
        for _ in range(3):
            await service._record_failure()
        assert service._circuit_open is True

    @pytest.mark.asyncio
    async def test_record_failure_below_threshold(self, service):
        service._max_failures = 5
        await service._record_failure()
        assert service._circuit_open is False

    def test_is_circuit_open(self, service):
        assert service.is_circuit_open() is False
        service._circuit_open = True
        assert service.is_circuit_open() is True


# ===========================================================================
# Core Cache Operations: get / set / delete / exists
# ===========================================================================


class TestCoreOps:
    @pytest.mark.asyncio
    async def test_get_returns_deserialized(self, service):
        service._client.get = AsyncMock(return_value=json.dumps({"foo": "bar"}))
        result = await service.get("key1")
        assert result == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, service):
        service._client.get = AsyncMock(return_value=None)
        assert await service.get("missing") is None

    @pytest.mark.asyncio
    async def test_get_returns_raw_on_bad_json(self, service):
        service._client.get = AsyncMock(return_value="not-json")
        result = await service.get("key")
        assert result == "not-json"

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disconnected(self, service):
        service._connected = False
        assert await service.get("key") is None

    @pytest.mark.asyncio
    async def test_get_returns_none_when_circuit_open(self, service):
        service._circuit_open = True
        assert await service.get("key") is None

    @pytest.mark.asyncio
    async def test_get_records_failure_on_exception(self, service):
        service._client.get = AsyncMock(side_effect=Exception("boom"))
        result = await service.get("key")
        assert result is None
        assert service._failure_count == 1

    @pytest.mark.asyncio
    async def test_set_success(self, service):
        service._client.setex = AsyncMock()
        result = await service.set("k", {"v": 1}, ttl=60)
        assert result is True
        service._client.setex.assert_awaited_once_with("k", 60, json.dumps({"v": 1}))

    @pytest.mark.asyncio
    async def test_set_default_ttl(self, service):
        service._client.setex = AsyncMock()
        await service.set("k", "v")
        args = service._client.setex.call_args
        assert args[0][1] == CacheTTL.API_RESPONSE

    @pytest.mark.asyncio
    async def test_set_returns_false_when_disconnected(self, service):
        service._connected = False
        assert await service.set("k", "v") is False

    @pytest.mark.asyncio
    async def test_set_records_failure_on_exception(self, service):
        service._client.setex = AsyncMock(side_effect=Exception("err"))
        result = await service.set("k", "v")
        assert result is False
        assert service._failure_count == 1

    @pytest.mark.asyncio
    async def test_delete_success(self, service):
        service._client.delete = AsyncMock()
        result = await service.delete("k")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_disconnected(self, service):
        service._connected = False
        result = await service.delete("k")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_records_failure_on_exception(self, service):
        service._client.delete = AsyncMock(side_effect=Exception("err"))
        result = await service.delete("k")
        assert result is False
        assert service._failure_count == 1

    @pytest.mark.asyncio
    async def test_exists_true(self, service):
        service._client.exists = AsyncMock(return_value=1)
        assert await service.exists("k") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, service):
        service._client.exists = AsyncMock(return_value=0)
        assert await service.exists("k") is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_disconnected(self, service):
        service._connected = False
        assert await service.exists("k") is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_exception(self, service):
        service._client.exists = AsyncMock(side_effect=Exception("err"))
        assert await service.exists("k") is False


# ===========================================================================
# delete_pattern
# ===========================================================================


class TestDeletePattern:
    @pytest.mark.asyncio
    async def test_delete_pattern_deletes_matching_keys(self, service):
        async def fake_scan(*_a, **_kw):
            for k in ["dtc:code:P0300", "dtc:code:P0301"]:
                yield k

        service._client.scan_iter = fake_scan
        service._client.delete = AsyncMock(return_value=2)
        result = await service.delete_pattern("dtc:code:*")
        assert result == 2

    @pytest.mark.asyncio
    async def test_delete_pattern_no_keys(self, service):
        async def empty_scan(*_a, **_kw):
            return
            yield  # pragma: no cover - unreachable, makes async generator

        service._client.scan_iter = empty_scan
        result = await service.delete_pattern("nope:*")
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_pattern_returns_zero_when_disconnected(self, service):
        service._connected = False
        assert await service.delete_pattern("*") == 0


# ===========================================================================
# invalidate_diagnosis_cache
# ===========================================================================


class TestInvalidateDiagnosisCache:
    @pytest.mark.asyncio
    async def test_invalidate_calls_delete_pattern(self, service):
        with patch.object(
            service, "delete_pattern", new_callable=AsyncMock, return_value=3
        ) as mock_dp:
            result = await service.invalidate_diagnosis_cache("sess1", "user1")
            assert result == 6  # 3 per pattern x 2 patterns
            assert mock_dp.await_count == 2


# ===========================================================================
# Batch Operations: mget / mset
# ===========================================================================


class TestBatchOps:
    @pytest.mark.asyncio
    async def test_mget_success(self, service):
        service._client.mget = AsyncMock(
            return_value=[json.dumps({"a": 1}), None, json.dumps([1, 2])]
        )
        result = await service.mget(["k1", "k2", "k3"])
        assert result == [{"a": 1}, None, [1, 2]]

    @pytest.mark.asyncio
    async def test_mget_empty_keys(self, service):
        result = await service.mget([])
        assert result == []

    @pytest.mark.asyncio
    async def test_mget_returns_nones_when_disconnected(self, service):
        service._connected = False
        result = await service.mget(["a", "b"])
        assert result == [None, None]

    @pytest.mark.asyncio
    async def test_mget_records_failure_on_exception(self, service):
        service._client.mget = AsyncMock(side_effect=Exception("err"))
        result = await service.mget(["a"])
        assert result == [None]
        assert service._failure_count == 1

    @pytest.mark.asyncio
    async def test_mset_success(self, service):
        pipe_mock = AsyncMock()
        pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
        pipe_mock.__aexit__ = AsyncMock(return_value=False)
        service._client.pipeline = MagicMock(return_value=pipe_mock)

        result = await service.mset({"k1": "v1", "k2": "v2"}, ttl=120)
        assert result is True
        assert pipe_mock.setex.call_count == 2

    @pytest.mark.asyncio
    async def test_mset_empty_mapping(self, service):
        assert await service.mset({}) is False

    @pytest.mark.asyncio
    async def test_mset_returns_false_when_disconnected(self, service):
        service._connected = False
        assert await service.mset({"k": "v"}) is False

    @pytest.mark.asyncio
    async def test_mset_records_failure_on_exception(self, service):
        pipe_mock = AsyncMock()
        pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
        pipe_mock.__aexit__ = AsyncMock(return_value=False)
        pipe_mock.execute = AsyncMock(side_effect=Exception("err"))
        service._client.pipeline = MagicMock(return_value=pipe_mock)

        assert await service.mset({"k": "v"}) is False
        assert service._failure_count == 1


# ===========================================================================
# DTC Code Cache helpers
# ===========================================================================


class TestDTCCodeCache:
    @pytest.mark.asyncio
    async def test_get_dtc_code(self, service):
        with patch.object(
            service, "get", new_callable=AsyncMock, return_value={"code": "P0300"}
        ) as mock_get:
            result = await service.get_dtc_code("p0300")
            assert result == {"code": "P0300"}
            mock_get.assert_awaited_once_with("dtc:code:P0300")

    @pytest.mark.asyncio
    async def test_set_dtc_code(self, service):
        with patch.object(service, "set", new_callable=AsyncMock, return_value=True) as mock_set:
            result = await service.set_dtc_code("p0300", {"desc": "test"})
            assert result is True
            mock_set.assert_awaited_once_with("dtc:code:P0300", {"desc": "test"}, CacheTTL.DTC_CODE)

    @pytest.mark.asyncio
    async def test_get_related_codes(self, service):
        with patch.object(service, "get", new_callable=AsyncMock, return_value=[]) as mock_get:
            await service.get_related_codes("P0301")
            mock_get.assert_awaited_once_with("dtc:related:P0301")

    @pytest.mark.asyncio
    async def test_set_related_codes(self, service):
        with patch.object(service, "set", new_callable=AsyncMock, return_value=True) as mock_set:
            await service.set_related_codes("P0301", [{"code": "P0300"}])
            mock_set.assert_awaited_once_with(
                "dtc:related:P0301", [{"code": "P0300"}], CacheTTL.DTC_CODE
            )


# ===========================================================================
# DTC Search Cache & key generation
# ===========================================================================


class TestDTCSearchCache:
    def test_make_search_key_deterministic(self, service):
        key1 = service._make_search_key("engine misfire", "powertrain", 10)
        key2 = service._make_search_key("engine misfire", "powertrain", 10)
        assert key1 == key2
        assert key1.startswith(CachePrefix.DTC_SEARCH)

    def test_make_search_key_case_insensitive(self, service):
        assert service._make_search_key("ABC", None, 20) == service._make_search_key(
            "abc", None, 20
        )

    def test_make_search_key_different_params(self, service):
        k1 = service._make_search_key("test", None, 10)
        k2 = service._make_search_key("test", "cat", 10)
        assert k1 != k2

    @pytest.mark.asyncio
    async def test_get_dtc_search_results(self, service):
        with patch.object(service, "get", new_callable=AsyncMock, return_value=[{"code": "P0300"}]):
            result = await service.get_dtc_search_results("misfire", "powertrain", 5)
            assert result == [{"code": "P0300"}]

    @pytest.mark.asyncio
    async def test_set_dtc_search_results(self, service):
        with patch.object(service, "set", new_callable=AsyncMock, return_value=True) as mock_set:
            await service.set_dtc_search_results("misfire", [{"code": "P0300"}], "powertrain", 5)
            mock_set.assert_awaited_once()
            call_args = mock_set.call_args
            assert call_args[0][2] == CacheTTL.DTC_SEARCH


# ===========================================================================
# NHTSA Data Cache
# ===========================================================================


class TestNHTSACache:
    @pytest.mark.asyncio
    async def test_get_nhtsa_recalls(self, service):
        with patch.object(service, "get", new_callable=AsyncMock, return_value=[]) as mock_get:
            await service.get_nhtsa_recalls("VW", "Golf", 2018)
            mock_get.assert_awaited_once_with("nhtsa:recalls:VW:Golf:2018")

    @pytest.mark.asyncio
    async def test_set_nhtsa_recalls(self, service):
        with patch.object(service, "set", new_callable=AsyncMock, return_value=True) as mock_set:
            await service.set_nhtsa_recalls("VW", "Golf", 2018, [{"id": 1}])
            mock_set.assert_awaited_once_with(
                "nhtsa:recalls:VW:Golf:2018", [{"id": 1}], CacheTTL.NHTSA_DATA
            )

    @pytest.mark.asyncio
    async def test_get_nhtsa_complaints(self, service):
        with patch.object(service, "get", new_callable=AsyncMock, return_value=None) as mock_get:
            await service.get_nhtsa_complaints("Toyota", "Corolla", 2020)
            mock_get.assert_awaited_once_with("nhtsa:complaints:Toyota:Corolla:2020")

    @pytest.mark.asyncio
    async def test_set_nhtsa_complaints(self, service):
        with patch.object(service, "set", new_callable=AsyncMock, return_value=True) as mock_set:
            await service.set_nhtsa_complaints("Toyota", "Corolla", 2020, [])
            mock_set.assert_awaited_once_with(
                "nhtsa:complaints:Toyota:Corolla:2020", [], CacheTTL.NHTSA_DATA
            )

    @pytest.mark.asyncio
    async def test_get_vin_decode(self, service):
        with patch.object(
            service, "get", new_callable=AsyncMock, return_value={"make": "VW"}
        ) as mock_get:
            result = await service.get_vin_decode("wvwzzz1kz1234")
            assert result == {"make": "VW"}
            mock_get.assert_awaited_once_with("nhtsa:vin:WVWZZZ1KZ1234")

    @pytest.mark.asyncio
    async def test_set_vin_decode(self, service):
        with patch.object(service, "set", new_callable=AsyncMock, return_value=True) as mock_set:
            await service.set_vin_decode("wvwzzz1kz1234", {"make": "VW"})
            mock_set.assert_awaited_once_with(
                "nhtsa:vin:WVWZZZ1KZ1234", {"make": "VW"}, CacheTTL.NHTSA_DATA
            )


# ===========================================================================
# Embedding Cache
# ===========================================================================


class TestEmbeddingCache:
    @pytest.mark.asyncio
    async def test_get_embedding(self, service):
        with patch.object(
            service, "get", new_callable=AsyncMock, return_value=[0.1, 0.2]
        ) as mock_get:
            result = await service.get_embedding("motor hibakod")
            assert result == [0.1, 0.2]
            expected_hash = hashlib.md5(b"motor hibakod", usedforsecurity=False).hexdigest()
            mock_get.assert_awaited_once_with(f"embed:{expected_hash}")

    @pytest.mark.asyncio
    async def test_set_embedding(self, service):
        with patch.object(service, "set", new_callable=AsyncMock, return_value=True) as mock_set:
            await service.set_embedding("test", [0.1])
            expected_hash = hashlib.md5(b"test", usedforsecurity=False).hexdigest()
            mock_set.assert_awaited_once_with(f"embed:{expected_hash}", [0.1], CacheTTL.EMBEDDINGS)

    @pytest.mark.asyncio
    async def test_get_embeddings_batch(self, service):
        with patch.object(
            service, "mget", new_callable=AsyncMock, return_value=[[0.1], None]
        ) as mock_mget:
            result = await service.get_embeddings_batch(["text1", "text2"])
            assert result == [[0.1], None]
            assert mock_mget.await_count == 1


# ===========================================================================
# Rate Limiting
# ===========================================================================


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self, service):
        pipe_mock = AsyncMock()
        pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
        pipe_mock.__aexit__ = AsyncMock(return_value=False)
        pipe_mock.execute = AsyncMock(return_value=[3, True])  # 3rd request
        service._client.pipeline = MagicMock(return_value=pipe_mock)

        allowed, remaining = await service.check_rate_limit("ip:1.2.3.4", 10, 60)
        assert allowed is True
        assert remaining == 7

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, service):
        pipe_mock = AsyncMock()
        pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
        pipe_mock.__aexit__ = AsyncMock(return_value=False)
        pipe_mock.execute = AsyncMock(return_value=[11, True])
        service._client.pipeline = MagicMock(return_value=pipe_mock)

        allowed, remaining = await service.check_rate_limit("ip:1.2.3.4", 10, 60)
        assert allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_rate_limit_fail_closed_when_disconnected(self, service):
        service._connected = False
        allowed, remaining = await service.check_rate_limit("ip:1.2.3.4", 10, 60)
        assert allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_rate_limit_fail_closed_on_exception(self, service):
        pipe_mock = AsyncMock()
        pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
        pipe_mock.__aexit__ = AsyncMock(return_value=False)
        pipe_mock.execute = AsyncMock(side_effect=Exception("redis down"))
        service._client.pipeline = MagicMock(return_value=pipe_mock)

        allowed, remaining = await service.check_rate_limit("ip:1.2.3.4", 10, 60)
        assert allowed is False
        assert remaining == 0


# ===========================================================================
# Statistics
# ===========================================================================


class TestStats:
    @pytest.mark.asyncio
    async def test_get_stats_disconnected(self, service):
        service._connected = False
        stats = await service.get_stats()
        assert stats == {"status": "disconnected"}

    @pytest.mark.asyncio
    async def test_get_stats_connected(self, service):
        service._client.info = AsyncMock(
            return_value={
                "used_memory_human": "1.5M",
                "connected_clients": 5,
                "keyspace_hits": 80,
                "keyspace_misses": 20,
            }
        )
        service._client.dbsize = AsyncMock(return_value=42)

        stats = await service.get_stats()
        assert stats["status"] == "connected"
        assert stats["total_keys"] == 42
        assert stats["hit_rate"] == 80.0

    @pytest.mark.asyncio
    async def test_get_stats_error(self, service):
        service._client.info = AsyncMock(side_effect=Exception("fail"))
        stats = await service.get_stats()
        assert stats["status"] == "error"

    def test_calculate_hit_rate_zero_total(self, service):
        assert service._calculate_hit_rate({}) == 0.0

    def test_calculate_hit_rate_normal(self, service):
        info = {"keyspace_hits": 75, "keyspace_misses": 25}
        assert service._calculate_hit_rate(info) == 75.0


# ===========================================================================
# @cached decorator
# ===========================================================================


class TestCachedDecorator:
    @pytest.mark.asyncio
    async def test_cached_returns_from_cache_on_hit(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value={"cached": True})

        with patch("app.db.redis_cache.get_cache_service", return_value=mock_cache):

            @cached("test:", ttl=60)
            async def my_func(x: int) -> dict:
                return {"computed": True}

            result = await my_func(1)
            assert result == {"cached": True}

    @pytest.mark.asyncio
    async def test_cached_computes_on_miss(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        with patch("app.db.redis_cache.get_cache_service", return_value=mock_cache):

            @cached("test:", ttl=60)
            async def my_func(x: int) -> dict:
                return {"computed": True}

            result = await my_func(1)
            assert result == {"computed": True}
            mock_cache.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cached_with_custom_key_builder(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        with patch("app.db.redis_cache.get_cache_service", return_value=mock_cache):

            @cached("dtc:", ttl=60, key_builder=lambda code: code.upper())
            async def lookup(code: str) -> dict:
                return {"code": code}

            result = await lookup("p0300")
            assert result == {"code": "p0300"}
            # Verify key was built with key_builder
            get_key = mock_cache.get.call_args[0][0]
            assert get_key == "dtc:P0300"

    @pytest.mark.asyncio
    async def test_cached_graceful_on_cache_error(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(side_effect=Exception("redis down"))
        mock_cache.set = AsyncMock(side_effect=Exception("redis down"))

        with patch("app.db.redis_cache.get_cache_service", return_value=mock_cache):

            @cached("test:", ttl=60)
            async def my_func() -> str:
                return "ok"

            result = await my_func()
            assert result == "ok"  # function still works despite cache errors
