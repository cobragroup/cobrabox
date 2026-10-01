from cobrabox import feature
from cobrabox.egg.alignments import ALIGNMENTS


def test_alignment_table_matches_registry() -> None:
    """
    Every registered feature has an alignment, and no stale entries remain.
    """
    registered = {
        n
        for n in dir(feature)
        if isinstance(getattr(feature, n), type)
        and getattr(getattr(feature, n), "_is_cobrabox_feature", False)
    }
    assert set(ALIGNMENTS) == registered
