def test_confidence_threshold_logic():
    threshold = 0.70
    assert 0.91 >= threshold
    assert not (0.42 >= threshold)
