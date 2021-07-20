import pytest

from ..util import *


def test_pct_change():
    with pytest.raises(ZeroDivisionError):
        pct_change(0, 1)

    assert pct_change(1, 0) == -100
    assert pct_change(1, 2) == 100

    assert pct_change(1, 1) == 0
