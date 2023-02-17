import pytest
import numpy as np
from api.lib.autograd.node import Variable, Placeholder


def test_run_forward_no_placeholder(session, test_case_binary):
    x_val, y_val = test_case_binary

    x = Variable(x_val)
    y = Variable(y_val)

    x_val, y_val = np.asarray(x_val), np.asarray(y_val)
    assert np.array_equal(session.run(x), x_val)
    assert np.array_equal(session.run(y-x), y_val-x_val)


def test_run_forward_with_placeholder(session, test_case_binary):
    x_val, y_val = test_case_binary

    x = Variable(x_val)
    y = Placeholder('w')

    with pytest.raises(KeyError):
        session.run(x * y)

    with pytest.raises(KeyError):
        session.run(x * y, feed_dict={'y': (y for y in [y_val])})

    z = session.run(x * y, feed_dict={'w': (y for y in [y_val, y_val])})
    x_val, y_val = np.asarray(x_val), np.asarray(y_val)
    assert np.array_equal(z, [y_val*x_val, y_val*x_val])


def test_run_forward_multiple_head_nodes(session, test_case_binary):
    x_val, y_val = test_case_binary

    x = Variable(x_val)
    y = Variable(y_val)

    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)

    add_op = x + y
    mul_op = x * y
    add_1_op = add_op + mul_op
    useless = x * add_op

    add_1_out, useless_out = session.run([add_1_op, useless])

    assert np.array_equal(add_1_out, (x_val + y_val) + (y_val * x_val))
    assert np.array_equal(
        useless_out,
        x_val * (x_val + y_val)
    )


def test_run_backward_no_placeholder(graph, session, test_case_binary):

    x_val, y_val = test_case_binary
    x, y = Variable(x_val, name='x'), Variable(y_val, name='y')
    out = 2*x + 3*x*y

    frwrd = session.run(out)  # fill Operator nodes with a value
    grads = session.gradients(out)

    x_val, y_val = np.asarray(x_val), np.asarray(y_val)
    assert np.array_equal(2 * x_val + 3*x_val*y_val, frwrd)
    assert np.array_equal(2 + 3*y_val, grads[x])
    assert np.array_equal(3*x_val, grads[y])


def test_run_backward_with_placeholder(session):
    w = Variable(1, name='w')
    x, x_val = Placeholder(name='x'), 2
    op = w*x

    with pytest.raises(TypeError):
        session.gradients(op)

    x.value = x_val
    grd = session.gradients(op)
    assert grd[x] == w.value and grd[w] == x_val
