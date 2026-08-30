import pytest


@pytest.fixture(autouse=True)
def _isolated_trace_dir(tmp_path, monkeypatch):
    """Keep test runs hermetic: traces go to tmp_path, not the global tmpdir."""
    monkeypatch.setattr("config.settings.SEISMIC_TRACE_DIR", str(tmp_path / "traces"))


class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = FakeFunc(name, arguments)


class FakeLLMClient:
    """Returns scripted completions; no network. Records each call's kwargs
    so tests can assert on the messages actually sent to the model."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get_completion(self, *a, **k):
        self.calls.append(k)
        return self._responses.pop(0)


@pytest.fixture
def fake_llm_factory():
    return lambda responses: FakeLLMClient(responses)


class FakeVisionClient:
    """Scripted vision responses (raw text); records (mime, prompt) per call."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def interpret_image(self, image_bytes, mime, prompt):
        self.calls.append((mime, prompt))
        if not self._responses:
            raise AssertionError("FakeVisionClient: no scripted response left")
        return self._responses.pop(0)


@pytest.fixture
def fake_vision_factory():
    return lambda responses: FakeVisionClient(responses)


@pytest.fixture
def outcrop_image(tmp_path):
    """400x200 synthetic 'outcrop': grey background, dark band rows 40-80, pale lens."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (400, 200), (150, 140, 130))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 40, 399, 80], fill=(60, 55, 50))
    d.ellipse([100, 110, 300, 170], fill=(220, 210, 190))
    path = tmp_path / "outcrop.png"
    im.save(path)
    return str(path)
