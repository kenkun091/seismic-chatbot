from interfaces.security import RateLimiter, check_api_key


def test_check_api_key_accepts_match():
    assert check_api_key("secret", expected="secret") is True


def test_check_api_key_rejects_mismatch():
    assert check_api_key("wrong", expected="secret") is False


def test_check_api_key_rejects_missing():
    assert check_api_key(None, expected="secret") is False


def test_rate_limiter_allows_under_limit():
    rl = RateLimiter(max_requests=2, window_seconds=60)
    assert rl.allow("ip1", now=0.0) is True
    assert rl.allow("ip1", now=1.0) is True


def test_rate_limiter_blocks_over_limit():
    rl = RateLimiter(max_requests=2, window_seconds=60)
    rl.allow("ip1", now=0.0)
    rl.allow("ip1", now=1.0)
    assert rl.allow("ip1", now=2.0) is False


def test_rate_limiter_resets_after_window():
    rl = RateLimiter(max_requests=2, window_seconds=60)
    rl.allow("ip1", now=0.0)
    rl.allow("ip1", now=1.0)
    assert rl.allow("ip1", now=61.0) is True


def test_rate_limiter_isolates_keys():
    rl = RateLimiter(max_requests=1, window_seconds=60)
    assert rl.allow("ip1", now=0.0) is True
    assert rl.allow("ip2", now=0.0) is True
    assert rl.allow("ip1", now=0.0) is False
