import pytest
from api.lib import namespace, autograd as ag
from api.lib.autograd import utils


def test_form_feed_dict():
    x = namespace.nodes.placeholder('x')
    y = namespace.nodes.placeholder('y')

    formed = utils.form_feed_dict([['x_val']], x)
    assert next(formed['x']) == 'x_val'

    formed = utils.form_feed_dict([[['x_val'], ['x_val_1']], ['y_val']], x, y)
    assert next(formed['x']) == 'x_val'
    assert next(formed['x']) == 'x_val_1'
    with pytest.raises(StopIteration):
        next(formed['x'])
    assert next(formed['y']) == 'y_val'

    with pytest.raises(ValueError):
        utils.form_feed_dict([], x)

    with pytest.raises(ValueError):
        utils.form_feed_dict([1, 2, 3], x)

    with pytest.raises(ValueError):
        utils.form_feed_dict([1, 2, 3], x, y)


def test_convert_to_node(session):
    test_case_var = utils.convert_to_tensor(
        'variable',
        value=0.01,
        name='test'
    )
    assert isinstance(test_case_var, ag.node.Variable)
    assert session.run(test_case_var) == 0.01 and test_case_var.name == 'test'

    test_case_pl = utils.convert_to_tensor(
        'placeholder',
        name='test'
    )
    assert isinstance(test_case_pl, ag.node.Placeholder)
    assert test_case_pl.value is None and test_case_pl.name == 'test'

    test_case_cnst = utils.convert_to_tensor(
        'constant',
        value=1,
        name='test'
    )
    assert isinstance(test_case_cnst, ag.node.Constant)
    assert test_case_cnst.value == 1 and test_case_cnst.name == 'test'


def test_fill_placeholders():
    x_pl = ag.node.Placeholder('x')
    y_pl = ag.node.Placeholder('y')

    pls = [
        ag.node.Variable(1, name='x'),
        x_pl,
        y_pl,
        'not-placeholder'
    ]

    x_val = 1
    utils.fill_placeholders(*pls, feed_dict={'x': x_val})
    assert x_pl.value == x_val
    assert y_pl.value is None

