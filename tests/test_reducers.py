import operator
import pytest
from agents.researcher.models import override_reducer


def test_override_reducer_none_handling():
    # Both None
    assert override_reducer(None, None) == []
    
    # current_value None, new_value list
    assert override_reducer(None, ["note1", "note2"]) == ["note1", "note2"]
    
    # current_value list, new_value None
    assert override_reducer(["note1"], None) == ["note1"]


def test_override_reducer_list_addition():
    # Regular list concatenation
    result = override_reducer(["a", "b"], ["c", "d"])
    assert result == ["a", "b", "c", "d"]


def test_override_reducer_override_dict():
    # Override when current value exists
    override_payload = {"type": "override", "value": ["new_state_only"]}
    result = override_reducer(["old1", "old2"], override_payload)
    assert result == ["new_state_only"]

    # Override when current value is None
    result_none = override_reducer(None, override_payload)
    assert result_none == ["new_state_only"]


def test_override_reducer_raw_override():
    # Override without value field returns the dict
    override_payload = {"type": "override"}
    result = override_reducer(["old"], override_payload)
    assert result == override_payload


def test_override_reducer_scalar_append_dict_merge_and_none_value():
    # Appending a scalar to a list
    assert override_reducer(["a", "b"], "c") == ["a", "b", "c"]

    # Merging dicts
    assert override_reducer({"x": 1}, {"y": 2}) == {"x": 1, "y": 2}

    # Override with explicit None value returns []
    assert override_reducer(["old"], {"type": "override", "value": None}) == []

    # Incompatible types fallback to new_value
    assert override_reducer(42, "string") == "string"
