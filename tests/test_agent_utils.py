import json
import importlib
import pytest
from unittest.mock import patch, MagicMock

agent_101 = importlib.import_module("agents.101.agent")
get_weather = getattr(agent_101, "get_weather")
get_user_preferences = getattr(agent_101, "get_user_preferences")
book_recommendation = getattr(agent_101, "book_recommendation")

from utils.utils import get_engine_for_chinook_db


def test_user_preferences():
    assert "sci-fi" in get_user_preferences.invoke("alice").lower()
    assert "comedy" in get_user_preferences.invoke("bob").lower()
    assert get_user_preferences.invoke("unknown_user") == "No preferences found"


def test_book_recommendation():
    sci_fi = book_recommendation.invoke({"genre": "sci-fi"})
    assert "Arrival" in sci_fi
    comedy = book_recommendation.invoke({"genre": "comedy"})
    assert "Lebowski" in comedy
    unknown = book_recommendation.invoke({"genre": "western"})
    assert unknown == "No recommendations available"


def test_get_weather_coordinate_validation():
    # Invalid latitude
    res_lat = json.loads(get_weather.invoke({"latitude": 95.0, "longitude": 0.0}))
    assert "error" in res_lat
    assert "Invalid latitude" in res_lat["error"]

    # Invalid longitude
    res_lon = json.loads(get_weather.invoke({"latitude": 10.0, "longitude": 185.0}))
    assert "error" in res_lon
    assert "Invalid longitude" in res_lon["error"]


def test_get_weather_mocked_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "current": {
            "temperature_2m": 72.5,
            "weather_code": 1
        }
    }
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        res = json.loads(get_weather.invoke({"latitude": 37.77, "longitude": -122.41}))
        assert res["temperature_fahrenheit"] == 72.5
        assert res["weather_code"] == 1


def test_get_engine_for_chinook_db_custom_script():
    sql = "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT); INSERT INTO test VALUES (1, 'GraphForge');"
    engine = get_engine_for_chinook_db(sql_script=sql)
    with engine.connect() as conn:
        result = conn.exec_driver_sql("SELECT name FROM test WHERE id = 1").fetchone()
        assert result[0] == "GraphForge"
