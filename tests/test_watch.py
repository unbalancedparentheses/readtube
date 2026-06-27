"""Tests for watch: channel-primary discovery + title-substring creator match.

No network — fetch_channel_recent / search_query / run_watch are not exercised.
"""

from __future__ import annotations

import pytest

from readtube.watch import (
    Channel,
    Creator,
    WatchConfig,
    _channel_url,
    _normalize_handle,
    append_seen,
    discover,
    load_seen,
    match_creator,
)


def _entry(vid: str, title: str, duration: float = 1800.0, **extra) -> dict:
    e = {"id": vid, "title": title, "duration": duration}
    e.update(extra)
    return e


CREATORS = [
    Creator(name="Lyn Alden", aliases=["lyn alden"]),
    Creator(name="Arthur Hayes", aliases=["arthur hayes"]),
    Creator(name="Druckenmiller", aliases=["druckenmiller"]),
]


class TestMatchCreator:
    def test_substring_match(self):
        c = match_creator("Bitcoin & Macro with Lyn Alden", CREATORS)
        assert c is not None and c.name == "Lyn Alden"

    def test_case_insensitive(self):
        c = match_creator("LYN ALDEN on rates", CREATORS)
        assert c is not None and c.name == "Lyn Alden"

    def test_no_match_returns_none(self):
        assert match_creator("Some other topic entirely", CREATORS) is None

    def test_empty_title(self):
        assert match_creator("", CREATORS) is None

    def test_misspelling_does_not_match(self):
        # Catches the "Atthur Hayes" concert-film impersonator case
        assert match_creator("Atthur Hayes Concert Film", CREATORS) is None

    def test_first_match_wins(self):
        creators = [
            Creator(name="Hayes", aliases=["hayes"]),
            Creator(name="Arthur", aliases=["arthur"]),
        ]
        c = match_creator("Arthur Hayes interview", creators)
        assert c is not None and c.name == "Hayes"

    def test_multiple_aliases(self):
        creators = [Creator(name="El-Erian", aliases=["el-erian", "el erian"])]
        assert match_creator("Mohamed El Erian on Fed", creators).name == "El-Erian"
        assert match_creator("El-Erian: outlook", creators).name == "El-Erian"


class TestChannelURL:
    def test_handle_prefix(self):
        assert _channel_url("@Foo") == "https://www.youtube.com/@Foo/videos"

    def test_bare_name(self):
        assert _channel_url("Foo") == "https://www.youtube.com/@Foo/videos"

    def test_full_url_passthrough(self):
        url = "https://www.youtube.com/channel/UCabc/videos"
        assert _channel_url(url) == url


class TestNormalizeHandle:
    def test_handle_form(self):
        assert _normalize_handle("@MacroVoices") == "macrovoices"

    def test_url_form(self):
        assert _normalize_handle("https://www.youtube.com/@MacroVoices") == "macrovoices"

    def test_url_with_trailing(self):
        assert _normalize_handle("https://www.youtube.com/@MacroVoices/videos") == "macrovoices"


class TestDiscover:
    def test_routes_via_title_match(self, monkeypatch):
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@MacroVoices")],
        )

        fake_entries = [
            _entry("v1", "MacroVoices #525 Lyn Alden: Inflation"),
            _entry("v2", "MacroVoices #524 generic week-in-review"),
            _entry("v3", "MacroVoices #523 Druckenmiller on Fed"),
        ]
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: fake_entries,
        )

        results = discover(cfg, seen=set(), verbose=False)
        ids = {entry["id"]: creator.name for creator, entry, _ in results}
        assert ids == {"v1": "Lyn Alden", "v3": "Druckenmiller"}

    def test_default_creator_routes_every_video(self, monkeypatch):
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@LynAldenMedia", default_creator="Lyn Alden")],
        )

        fake_entries = [
            _entry("v1", "Some random title with no name in it"),
            _entry("v2", "Another untitled topic"),
        ]
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: fake_entries,
        )

        results = discover(cfg, seen=set(), verbose=False)
        assert len(results) == 2
        assert all(creator.name == "Lyn Alden" for creator, _, _ in results)

    def test_seen_dedupes(self, monkeypatch):
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@C1")],
        )
        fake_entries = [_entry("v1", "Lyn Alden talk")]
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: fake_entries,
        )
        results = discover(cfg, seen={"v1"}, verbose=False)
        assert results == []

    def test_cross_channel_dedupe(self, monkeypatch):
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@C1"), Channel(handle="@C2")],
        )
        # Same id appears on both channels (reupload / cross-post)
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: [_entry("v1", "Lyn Alden interview")],
        )
        results = discover(cfg, seen=set(), verbose=False)
        assert len(results) == 1

    def test_filters_shorts_below_min_duration(self, monkeypatch):
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@C1")],
            min_duration=600,
        )
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: [
                _entry("v1", "Lyn Alden short clip", duration=120.0),
                _entry("v2", "Lyn Alden full episode", duration=3600.0),
            ],
        )
        results = discover(cfg, seen=set(), verbose=False)
        assert [e["id"] for _, e, _ in results] == ["v2"]

    def test_min_duration_applies_to_default_creator(self, monkeypatch):
        # The shorts on Ray Dalio's own channel were the motivating case
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@RD", default_creator="Druckenmiller")],
            min_duration=600,
        )
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: [
                _entry("v1", "30-second clip", duration=30.0),
                _entry("v2", "Long-form interview", duration=4000.0),
            ],
        )
        results = discover(cfg, seen=set(), verbose=False)
        assert [e["id"] for _, e, _ in results] == ["v2"]

    def test_missing_duration_drops_entry(self, monkeypatch):
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@C1")],
            min_duration=600,
        )
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: [_entry("v1", "Lyn Alden", duration=None)],
        )
        assert discover(cfg, seen=set(), verbose=False) == []

    def test_duplicate_title_within_channel_dedupes(self, monkeypatch):
        # Macro Voices case: same episode re-uploaded with distinct ids, same title
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@MV")],
            min_duration=600,
        )
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: [
                _entry("v1", "MacroVoices #528 Druckenmiller: Topic", duration=3000.0),
                _entry("v2", "MacroVoices #528 Druckenmiller: Topic", duration=3000.0),
            ],
        )
        results = discover(cfg, seen=set(), verbose=False)
        assert len(results) == 1

    def test_duplicate_title_dedupe_is_case_insensitive(self, monkeypatch):
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@C")],
            min_duration=600,
        )
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: [
                _entry("v1", "Lyn Alden On Inflation", duration=3000.0),
                _entry("v2", "LYN ALDEN ON INFLATION", duration=3000.0),
            ],
        )
        assert len(discover(cfg, seen=set(), verbose=False)) == 1

    def test_default_creator_unknown_creator_drops(self, monkeypatch):
        # WatchConfig.load validates this, but discover() should also no-op safely
        cfg = WatchConfig(
            creators=CREATORS,
            channels=[Channel(handle="@C1", default_creator="Nobody")],
        )
        monkeypatch.setattr(
            "readtube.watch.fetch_channel_recent",
            lambda handle, limit, verbose=True: [_entry("v1", "anything")],
        )
        results = discover(cfg, seen=set(), verbose=False)
        assert results == []


class TestSeenFile:
    def test_load_missing_returns_empty(self, tmp_path):
        assert load_seen(tmp_path / "nope.txt") == set()

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "seen.txt"
        append_seen(["a", "b", "c"], path)
        append_seen(["d"], path)
        assert load_seen(path) == {"a", "b", "c", "d"}

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "seen.txt"
        path.write_text("a\n\n  \nb\n")
        assert load_seen(path) == {"a", "b"}


class TestWatchConfigLoad:
    def test_missing_file_raises(self, tmp_path):
        from readtube.errors import ReadtubeError
        with pytest.raises(ReadtubeError):
            WatchConfig.load(tmp_path / "missing.toml")

    def test_loads_creators_and_channels(self, tmp_path):
        path = tmp_path / "watch.toml"
        path.write_text(
            """
[defaults]
recent_per_channel = 10
mode = "tldr"

[[creators]]
name = "Lyn Alden"
aliases = ["lyn alden"]

[[creators]]
name = "Druckenmiller"
aliases = ["druckenmiller", "stan druckenmiller"]

[[channels]]
handle = "@LynAldenMedia"
default_creator = "Lyn Alden"

[[channels]]
handle = "@MacroVoices"
"""
        )
        cfg = WatchConfig.load(path)
        assert cfg.recent_per_channel == 10
        assert cfg.mode == "tldr"
        assert len(cfg.creators) == 2
        assert cfg.creators[1].aliases == ["druckenmiller", "stan druckenmiller"]
        assert len(cfg.channels) == 2
        assert cfg.channels[0].default_creator == "Lyn Alden"
        assert cfg.channels[1].default_creator is None

    def test_no_creators_raises(self, tmp_path):
        from readtube.errors import ReadtubeError
        path = tmp_path / "watch.toml"
        path.write_text('[[channels]]\nhandle = "@x"\n')
        with pytest.raises(ReadtubeError):
            WatchConfig.load(path)

    def test_no_channels_raises(self, tmp_path):
        from readtube.errors import ReadtubeError
        path = tmp_path / "watch.toml"
        path.write_text('[[creators]]\nname = "A"\naliases = ["a"]\n')
        with pytest.raises(ReadtubeError):
            WatchConfig.load(path)

    def test_default_creator_must_exist(self, tmp_path):
        from readtube.errors import ReadtubeError
        path = tmp_path / "watch.toml"
        path.write_text(
            """
[[creators]]
name = "Lyn Alden"
aliases = ["lyn alden"]

[[channels]]
handle = "@x"
default_creator = "Nobody Here"
"""
        )
        with pytest.raises(ReadtubeError):
            WatchConfig.load(path)

    def test_aliases_lowercased_on_load(self, tmp_path):
        path = tmp_path / "watch.toml"
        path.write_text(
            """
[[creators]]
name = "Lyn Alden"
aliases = ["Lyn ALDEN", "ALDEN"]

[[channels]]
handle = "@x"
"""
        )
        cfg = WatchConfig.load(path)
        assert cfg.creators[0].aliases == ["lyn alden", "alden"]
