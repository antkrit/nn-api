import pytest
from api.core import autograd as ag
from api.core.autograd import utils
from tests.utils import check_node_name_format


def test_form_feed_dict():
    x = ag.Placeholder('x')
    y = ag.Placeholder('y')

    formed = utils.form_feed_dict([['x_val']], x)
    assert next(formed[x.name]) == 'x_val'

    formed = utils.form_feed_dict([[['x_val'], ['x_val_1']], ['y_val']], x, y)
    assert next(formed[x.name]) == 'x_val'
    assert next(formed[x.name]) == 'x_val_1'
    with pytest.raises(StopIteration):
        next(formed[x.name])
    assert next(formed[y.name]) == 'y_val'

    with pytest.raises(ValueError):
        utils.form_feed_dict([], x)

    with pytest.raises(ValueError):
        utils.form_feed_dict([1, 2, 3], x)

    with pytest.raises(ValueError):
        utils.form_feed_dict([1, 2, 3], x, y)


def test_convert_to_node(session):
    test_var = ag.Variable(1)
    test_case_var = utils.convert_to_node(value=test_var, name='test')
    assert test_case_var is test_var

    test_case_var = utils.convert_to_node(value=0.01, name='test')
    assert isinstance(test_case_var, ag.Variable)
    assert session.run(test_case_var) == 0.01
    assert check_node_name_format(test_case_var)

    test_case_var = utils.convert_to_node(
        value=None,
        name='test',
        to_variable=True
    )
    assert isinstance(test_case_var, ag.Variable)
    assert session.run(test_case_var) is None
    assert check_node_name_format(test_case_var)

    test_case_pl = utils.convert_to_node(name='test')
    assert isinstance(test_case_pl, ag.Placeholder)
    assert test_case_pl.value is None
    assert check_node_name_format(test_case_pl)

    test_case_pl = utils.convert_to_node(
        value=1,
        name='test',
        to_placeholder=True
    )
    assert isinstance(test_case_pl, ag.Placeholder)
    assert test_case_pl.value == 1
    assert check_node_name_format(test_case_pl)

    test_case_cnst = utils.convert_to_node(
        value=1,
        name='test',
        to_constant=True
    )
    assert isinstance(test_case_cnst, ag.Constant)
    assert test_case_cnst.value == 1
    assert check_node_name_format(test_case_cnst)


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
    utils.fill_placeholders(*pls, feed_dict={x_pl.name: x_val})
    assert x_pl.value == x_val
    assert y_pl.value is None

