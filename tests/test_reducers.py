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
