"""
tests.unit.application.test_matching_config | Layer: TEST
`NexusMatcher.from_config` must actually honour the config it is handed.

The parameter existed, was typed `config_path`, and was never read. Nothing failed: the
matcher built cleanly and ran with default thresholds, so a team that had tuned
`auto_approve_threshold` down to 0.80 went on auto-approving at 0.87 with no signal. A
config that is silently ignored is worse than one that is absent, because the user
believes the setting took effect.
"""

from __future__ import annotations

import json
import sys

import pytest

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
    _load_matching_config,
)


class TestLoadMatchingConfig:
    def test_none_gives_calibrated_defaults(self):
        cfg = _load_matching_config(None)
        assert cfg.auto_approve_threshold == 0.87
        assert cfg.fusion_alpha == 0.90

    def test_instance_passes_through_unchanged(self):
        original = MatchingConfig(auto_approve_threshold=0.85)
        assert _load_matching_config(original) is original

    def test_json_file(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"auto_approve_threshold": 0.80, "results_per_field": 3}))
        cfg = _load_matching_config(p)
        assert cfg.auto_approve_threshold == 0.80
        assert cfg.results_per_field == 3
        assert cfg.fusion_alpha == 0.90  # untouched keys keep their defaults

    def test_toml_file(self, tmp_path):
        """TOML needs a parser on 3.10, where tomllib is not yet stdlib."""
        pytest.importorskip(
            "tomllib" if sys.version_info >= (3, 11) else "tomli",
            reason="no TOML parser on this interpreter",
        )
        p = tmp_path / "m.toml"
        p.write_text("auto_approve_threshold = 0.92\nreview_threshold = 0.4\n")
        cfg = _load_matching_config(p)
        assert cfg.auto_approve_threshold == 0.92
        assert cfg.review_threshold == 0.4

    def test_wrapped_matching_section(self, tmp_path):
        """One project file may hold several sections."""
        p = tmp_path / "project.json"
        p.write_text(json.dumps({"matching": {"auto_approve_threshold": 0.75}}))
        assert _load_matching_config(p).auto_approve_threshold == 0.75

    def test_unknown_key_raises_rather_than_being_dropped(self, tmp_path):
        """
        The whole point. A typo must be loud: every field here is a measured number, and
        silently discarding `auto_approve_treshold` leaves the user certain they raised
        the bar while the matcher keeps auto-approving at the default.
        """
        p = tmp_path / "typo.json"
        p.write_text(json.dumps({"auto_approve_treshold": 0.99}))
        with pytest.raises(ValueError, match="auto_approve_treshold"):
            _load_matching_config(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_matching_config(tmp_path / "nope.json")

    def test_a_string_path_is_accepted(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"review_threshold": 0.33}))
        assert _load_matching_config(str(p)).review_threshold == 0.33


class TestFromConfigAppliesIt:
    """The end that matters: the loaded config must reach the built matcher."""

    def test_instance_reaches_the_matcher(self):
        matcher = NexusMatcher.from_config(
            MatchingConfig(auto_approve_threshold=0.85, fusion_alpha=0.7)
        )
        assert matcher._config.auto_approve_threshold == 0.85
        assert matcher._config.fusion_alpha == 0.7

    def test_file_reaches_the_matcher(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"auto_approve_threshold": 0.80}))
        assert NexusMatcher.from_config(p)._config.auto_approve_threshold == 0.80

    def test_no_argument_still_works(self):
        assert NexusMatcher.from_config()._config.auto_approve_threshold == 0.87
