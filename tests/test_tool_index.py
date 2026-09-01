import pytest
from core.tool_index import ToolIndex, ToolCard, render_card
from core.tool_registry import REGISTRY, REGISTRY_BY_NAME, AUTO_PLOT


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    # module-scoped: building embeds ~30 cards once, not per test
    return ToolIndex(persist_directory=str(tmp_path_factory.mktemp("tool_index")))


def test_render_card_contains_name_description_and_params():
    spec = REGISTRY_BY_NAME["make_ricker"]
    card = render_card(spec)
    assert card.startswith("make_ricker: ")
    assert spec.description in card
    assert "frequency (number, required)" in card


def test_plot_tools_are_excluded(index):
    plot_targets = set(AUTO_PLOT.values())
    names = {c.name for c in index.search("plot a wavelet figure", top_k=10, threshold=-1.0)}
    assert names.isdisjoint(plot_targets)


def test_search_returns_relevant_ranked_cards(index):
    cards = index.search("create a ricker wavelet with a given frequency")
    assert cards, "on-topic query must never return empty"
    assert cards[0].score >= cards[-1].score
    assert "make_ricker" in {c.name for c in cards[:3]}


def test_search_always_returns_top3_even_below_threshold(index):
    cards = index.search("completely unrelated cooking recipe", threshold=0.99)
    assert len(cards) == 3  # top-3 floor; nothing beyond 3 clears 0.99


def test_population_is_idempotent(tmp_path):
    d = str(tmp_path / "idx")
    a = ToolIndex(persist_directory=d)
    count_a = a.collection.count()
    b = ToolIndex(persist_directory=d)  # second startup, same dir
    assert b.collection.count() == count_a


def test_stale_tools_are_deleted_on_repopulation(tmp_path):
    d = str(tmp_path / "idx")
    ToolIndex(persist_directory=d)  # full registry
    subset = [s for s in REGISTRY if s.name != "make_ricker"]
    rebuilt = ToolIndex(persist_directory=d, specs=subset)
    names = {c.name for c in rebuilt.search("ricker wavelet", top_k=10, threshold=-1.0)}
    assert "make_ricker" not in names
