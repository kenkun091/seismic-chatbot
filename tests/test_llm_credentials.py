import pytest

from core.llm_client import resolve_llm_credentials


def test_databricks_preferred_when_present():
    api_key, base_url = resolve_llm_credentials(
        deepseek_key="dk", deepseek_url="https://ds",
        databricks_token="dbt", databricks_url="https://db",
    )
    assert (api_key, base_url) == ("dbt", "https://db")


def test_deepseek_used_when_databricks_absent():
    api_key, base_url = resolve_llm_credentials(
        deepseek_key="dk", deepseek_url="https://ds",
        databricks_token=None, databricks_url=None,
    )
    assert (api_key, base_url) == ("dk", "https://ds")


def test_no_credentials_raises():
    with pytest.raises(RuntimeError):
        resolve_llm_credentials(None, None, None, None)


def test_deepseek_key_without_base_url_raises():
    with pytest.raises(RuntimeError):
        resolve_llm_credentials(
            deepseek_key="dk", deepseek_url=None,
            databricks_token=None, databricks_url=None,
        )
