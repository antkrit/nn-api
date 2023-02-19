from api.lib import namespace


def test_base_container():
    init_a_value, next_a_value = 1, 3

    container = namespace.Container(name='test', a=init_a_value, b=2)
    assert container['a'] == init_a_value and container['b'] == 2

    container['a'] = next_a_value
    assert container['a'] == next_a_value and container.a == next_a_value

    del container['b']
    assert 'b' not in container
    assert len(container) == 1

    assert container('a') == next_a_value

    class A:
        def __init__(self, a):
            self.a = a

        def __call__(self, *args, **kwargs):
            return self.a

    container['class_A'] = A
    assert container['class_A'] is A
    assert container('class_A') is A

    instance = container('class_A', compiled=True, a=init_a_value)
    assert instance is not A and isinstance(instance, A)
    assert instance.a == init_a_value
