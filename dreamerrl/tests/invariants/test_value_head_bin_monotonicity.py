def test_value_head_bin_monotonicity(net):
    """
    Value bins must be strictly increasing.
    """
    bins = net.make_bins()
    assert (bins[1:] > bins[:-1]).all()
