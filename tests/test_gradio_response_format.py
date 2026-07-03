"""The Gradio UI renders the {"reply", "images"} contract: reply text fills the
pending slot, then one history row per image (Gradio 3.x shows one file per
message)."""
from interfaces.gradio_interface import append_bot_response


def _pending(msg="q"):
    return [[msg, None]]


def test_reply_with_images_appends_one_row_per_image():
    out = append_bot_response(_pending(), {"reply": "Tuning is 12.5 m.",
                                           "images": ["/tmp/a.png", "/tmp/b.png"]})
    assert out[0] == ["q", "Tuning is 12.5 m."]
    assert out[1] == [None, ("/tmp/a.png",)]
    assert out[2] == [None, ("/tmp/b.png",)]


def test_reply_without_images():
    assert append_bot_response(_pending(), {"reply": "Hello", "images": []}) == [["q", "Hello"]]


def test_missing_images_key_tolerated():
    assert append_bot_response(_pending(), {"reply": "Hello"}) == [["q", "Hello"]]


def test_plain_string_response():
    assert append_bot_response(_pending(), "plain") == [["q", "plain"]]


def test_non_string_response_stringified():
    assert append_bot_response(_pending(), 42) == [["q", "42"]]
