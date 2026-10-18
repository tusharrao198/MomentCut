"""Filenames must fit the filesystem, which counts BYTES, not characters.

Regression cover for OSError 36 ("File name too long") killing /api/hook in
prod on 26-jul-2026: a Bengali video title passed the 100-CHARACTER cap while
already being ~300 bytes, and the derived temp name pushed it further.
"""
from urllib.parse import urlsplit

import pytest

main = pytest.importorskip("main")          # cv2 etc., absent in minimal CI
hooks = pytest.importorskip("hooks")

# ext4/APFS cap a single name at 255 bytes.
FS_LIMIT = 255
BENGALI = "মজার_ধাঁধা_শুনুন_আর_হাসুন_সকাল_হলেই_কার_মাথা_হারিয়ে_যায়কমেডি_ভিডিও"


class TestTruncateBytes:
    def test_short_text_untouched(self):
        assert main.truncate_bytes("hello", 100) == "hello"

    def test_trims_to_the_byte_budget(self):
        out = main.truncate_bytes(BENGALI * 5, 120)
        assert len(out.encode("utf-8")) <= 120

    def test_never_splits_a_multibyte_character(self):
        # Budget deliberately lands mid-character.
        for budget in range(1, 40):
            out = main.truncate_bytes(BENGALI, budget)
            out.encode("utf-8").decode("utf-8")   # must not raise
            assert len(out.encode("utf-8")) <= budget

    def test_ascii_is_unaffected_by_the_byte_rule(self):
        text = "a" * 200
        assert main.truncate_bytes(text, 120) == "a" * 120


class TestSanitizeFilename:
    def test_non_latin_title_fits_the_budget(self):
        out = main.sanitize_filename(BENGALI * 4)
        assert len(out.encode("utf-8")) <= main.MAX_TITLE_BYTES

    def test_leaves_room_for_every_decoration_the_pipeline_adds(self):
        # Worst case actually produced: a hook overlay built on top of an
        # auto-captioned clip of a long-titled video.
        stem = main.sanitize_filename(BENGALI * 4)
        clip = f"{stem}_clip_10.mp4"
        captioned = f"subtitled_1785060039_{clip}"
        hook_png = f"temp_hook_abcd1234_{captioned}.png"
        assert len(hook_png.encode("utf-8")) < FS_LIMIT, len(hook_png.encode("utf-8"))

    def test_replaces_unsafe_characters_with_underscores(self):
        assert main.sanitize_filename(
            'a<b>c:d"e/f\\g|h?i*j#k'
        ) == "a_b_c_d_e_f_g_h_i_j_k"
        assert main.sanitize_filename("with spaces") == "with_spaces"

    def test_replaces_url_delimiters_and_percent_encoding(self):
        safe = main.sanitize_filename("Mission #45? retry%23")
        assert safe == "Mission_45_retry_23"
        parsed = urlsplit(f"/videos/job/{safe}_clip_1.mp4")
        assert parsed.query == ""
        assert parsed.fragment == ""

    def test_collapses_replacement_runs_and_has_a_safe_fallback(self):
        assert main.sanitize_filename("a #?:% b") == "a_b"
        assert main.sanitize_filename("###") == "video"

    def test_local_upload_uses_the_same_safe_stem(self):
        # Docker presents the Windows bind mount as a normal Linux path.
        path = "/app/uploads/job_GTA Mission #45? retry%23.mp4"
        assert main.video_title_from_local_path(path) == (
            "job_GTA_Mission_45_retry_23"
        )

    def test_combining_accents_are_precomposed(self):
        """A Spanish title arrives from yt-dlp in either Unicode spelling.

        The decomposed form (o + U+0301) survived into the R2 key and the URL,
        where a fetch() of the object came back 503 while a <video> element
        loaded it fine — the download button broke on every accented title
        (24-ago-2026). Both spellings must produce the same, precomposed name.
        """
        decomposed = "cancio\u0301n"     # o + combining acute
        precomposed = "canci\u00f3n"     # single ó
        assert decomposed != precomposed
        assert main.sanitize_filename(decomposed) == precomposed
        assert main.sanitize_filename(precomposed) == precomposed
        assert "\u0301" not in main.sanitize_filename("Ha\u0301bitos_Nº1")

    def test_the_old_character_cap_would_have_overflowed(self):
        # Documents why the rule changed: 100 chars of Bengali is ~300 bytes.
        assert len((BENGALI * 4)[:100].encode("utf-8")) > FS_LIMIT


class TestHookTempName:
    def test_hook_helper_trims_by_bytes(self):
        out = hooks._truncate_bytes(BENGALI * 5, 80)
        assert len(out.encode("utf-8")) <= 80
        out.encode("utf-8").decode("utf-8")
