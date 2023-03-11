import pytest
import numpy as np
from api.lib import namespace


def test_run_forward_no_placeholder(session, test_case_binary):
    x_val, y_val = test_case_binary

    x = namespace.nodes.variable(x_val)
    y = namespace.nodes.variable(y_val)

    x_val, y_val = np.asarray(x_val), np.asarray(y_val)

    assert np.array_equal(session.run(x, feed_dict={'random': 'value'}), x_val)
    assert np.array_equal(session.run(y-x), y_val-x_val)

    assert np.array_equal(session.run(y-x, x, returns=[x]), x_val)


def test_run_forward_with_placeholder(session, test_case_binary):
    x_val, y_val = test_case_binary

    x = namespace.nodes.variable(x_val)
    y = namespace.nodes.placeholder('w')

    with pytest.raises(KeyError):
        session.run(x * y)

    with pytest.raises(KeyError):
        session.run(x * y, feed_dict={'y': iter([y_val])})

    z = session.run(x * y, feed_dict={y.name: iter([y_val, y_val])})
    x_val, y_val = np.asarray(x_val), np.asarray(y_val)
    assert np.array_equal(z, [y_val*x_val, y_val*x_val])


def test_run_forward_multiple_head_nodes(session, test_case_binary):
    x_val, y_val = test_case_binary

    x = namespace.nodes.variable(x_val)
    y = namespace.nodes.variable(y_val)

    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)

    add_op = x + y
    mul_op = x * y
    add_1_op = add_op + mul_op
    useless = x * add_op

    add_1_out, useless_out = session.run(add_1_op, useless)

    assert np.array_equal(add_1_out, (x_val + y_val) + (y_val * x_val))
    assert np.array_equal(
        useless_out,
        x_val * (x_val + y_val)
    )


def test_run_backward_no_placeholder(graph, session, test_case_binary):

    x_val, y_val = test_case_binary
    x = namespace.nodes.variable(x_val, name='x')
    y = namespace.nodes.variable(y_val, name='y')
    out = 2*x + 3*x*y

    session.run(out)  # fill Operator nodes with a value

    x_val, y_val = np.asarray(x_val), np.asarray(y_val)
    x_grd, y_grd = session.gradients(out, [x, y])
    assert np.array_equal(x_grd, 2 + 3*y_val)
    assert np.array_equal(y_grd, 3*x_val)
    assert session.ctx_get('gradients') is not None


def test_run_backward_with_placeholder(session):
    w = namespace.nodes.variable(1, name='w')
    x, x_val = namespace.nodes.placeholder(name='x'), 2

    op = w*x

    with pytest.raises(TypeError):
        session.gradients(op)

    x.value = x_val
    x_grd, w_grd = session.gradients(op, [x, w])
    assert x_grd == w.value and w_grd == x_val


def test_session_utils(session):
    assert session.ctx_get('globals') is not None

    some_token = 'some_token'
    some_token_value = 'test_value'
    session.ctx_add(some_token, some_token_value)
    assert session.ctx_get(some_token)[0] == some_token_value
    assert some_token in session.ctx_get('globals')

    var_value, pl_value = 1, 2
    variable = namespace.nodes.variable(var_value, name='test_variable')
    placeholder = namespace.nodes.placeholder(name='test_placeholder')
    operation = variable + placeholder

    feed_dict = {placeholder.name: iter([pl_value])}

    var = session._Session__process_node_forward(variable)
    assert var is variable and var.value == var_value

    with pytest.raises(KeyError):
        session._Session__process_node_forward(placeholder, feed_dict=None)

    with pytest.raises(KeyError):
        session._Session__process_node_forward(
            placeholder, feed_dict={'wrong_name': iter([pl_value])}
        )

    pl = session._Session__process_node_forward(
        placeholder, feed_dict=feed_dict
    )
    assert pl is placeholder and pl.value == pl_value

    with pytest.raises(StopIteration):
        session._Session__process_node_forward(
            placeholder, feed_dict=feed_dict
        )

    op = session._Session__process_node_forward(operation)
    assert op is operation and op.value == var_value + pl_value
