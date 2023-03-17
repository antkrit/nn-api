from api.core.autograd.graph import Graph, get_current_graph, reset_current_graph


def test_graph():
    gc = get_current_graph()
    assert gc is not None

    g = Graph()
    g.as_default()

    gc = get_current_graph()
    assert gc is g

    reset_current_graph()
    gc = get_current_graph()
    assert gc is not None and gc is not g

    with Graph() as g:
        gc = get_current_graph()
        gn = g.name
        assert gc is g

    gc = get_current_graph()
    assert gn != gc.name
