import pytest
from api.lib.autograd import utils, Placeholder


def test_form_feed_dict():
    x = Placeholder('x')
    y = Placeholder('y')

    formed = utils.form_feed_dict([['x_val']], x)
    assert next(formed['x']) == 'x_val'

    formed = utils.form_feed_dict([[['x_val'], ['x_val_1']], ['y_val']], x, y)
    assert next(formed['x']) == 'x_val'
    assert next(formed['x']) == 'x_val_1'
    assert next(formed['y']) == 'y_val'

    with pytest.raises(ValueError):
        utils.form_feed_dict([], x)

    with pytest.raises(ValueError):
        utils.form_feed_dict([1, 2, 3], x)

    with pytest.raises(ValueError):
        utils.form_feed_dict([1, 2, 3], x, y)
