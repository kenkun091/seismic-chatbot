from tools.parameter_linking import LinkType


def test_proportional_member_spelled_correctly():
    assert LinkType.PROPORTIONAL.value == "proportional"
    assert not hasattr(LinkType, "PROPIONAL")
