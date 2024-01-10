import pytest

from tw_experimentation.widgetizer import SampleSizeInterface


def test_sample_size_calculator():
    interface = SampleSizeInterface()
    interface.classical_test()


if __name__ == "__main__":
    pytest.main([__file__])
