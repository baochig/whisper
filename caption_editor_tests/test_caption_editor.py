from opencc import OpenCC

from caption_editor import (
    detect_format,
    replace_terms,
    convert_simplified_to_traditional,
    process_text,
)


def test_detect_format_srt():
    text = "1\n00:00:01,000 --> 00:00:02,000\nHello"
    assert detect_format(text) == "srt"


def test_detect_format_text():
    text = "Hello world"
    assert detect_format(text) == "text"


def test_replace_terms():
    assert replace_terms("AI and ML", {"AI": "人工智慧"}) == "人工智慧 and ML"


def test_convert_simplified_to_traditional():
    converter = OpenCC("s2t")
    assert convert_simplified_to_traditional("汉字", converter) == "漢字"


def test_process_text_chain():
    converter = OpenCC("s2t")
    result = process_text("AI漢字", {"AI": "人工智慧"}, converter)
    assert "人工智慧" in result and "漢字" in result
