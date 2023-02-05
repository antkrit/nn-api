import numpy as np
import pytest

from api.lib.preprocessing.initializers import *


NUMPY_SEED = np.random.randint(0, 100)


class TestBase:

    @pytest.mark.parametrize('si', [0, 0.01, 5])
    @pytest.mark.parametrize('mu', [0, 1])
    def test_normal_initializer(self, session, test_case_unary, si, mu):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        init_zeros = NormalInitializer()(size)
        assert np.array_equal(session.run(init_zeros), np.zeros(size))

        init_diff = NormalInitializer(mu=0, sigma=1)
        sample = session.run(init_diff(size))
        assert not np.array_equal(session.run(init_diff(size)), sample)

        np.random.seed(NUMPY_SEED)
        ndist = si*np.random.randn(*size) + mu

        init = NormalInitializer(mu=mu, sigma=si, seed=NUMPY_SEED)(size)
        assert np.array_equal(session.run(init), ndist)

    def test_uniform_initializer(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        init_zeros = UniformInitializer()(size)
        assert np.array_equal(session.run(init_zeros), np.zeros(size))

        init_diff = UniformInitializer(low=-1, high=1)
        sample = session.run(init_diff(size))
        assert not np.array_equal(session.run(init_diff(size)), sample)

        low_bnd = np.random.randint(-10, 0)
        high_bnd = np.random.randint(1, 10)
        np.random.seed(NUMPY_SEED)
        ndist = np.random.uniform(low_bnd, high_bnd, size=size)

        init = UniformInitializer(
            low=low_bnd,
            high=high_bnd,
            seed=NUMPY_SEED
        )(size)
        assert np.array_equal(session.run(init), ndist)


class TestNormalDistribution:

    def test_zeros(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        init = zeros(*size)
        assert np.array_equal(session.run(init), np.zeros(size))

    def test_ones(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        init = ones(*size)
        assert np.array_equal(session.run(init), np.ones(size))

    def test_random_normal(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        np.random.seed(NUMPY_SEED)
        ndist = np.random.randn(*size)

        init = random_normal(*size, seed=NUMPY_SEED)
        assert np.array_equal(session.run(init), ndist)

    def test_xavier(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        np.random.seed(NUMPY_SEED)
        ndist = np.random.randn(*size)
        expected = np.sqrt(2/(np.sum(size))) * ndist

        init = xavier_normal(*size, seed=NUMPY_SEED)
        assert np.array_equal(session.run(init), expected)

    def test_he(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        np.random.seed(NUMPY_SEED)
        ndist = np.random.randn(*size)
        expected = np.sqrt(2 / size[0]) * ndist

        init = he_normal(*size, seed=NUMPY_SEED)
        assert np.array_equal(session.run(init), expected)

    def test_lecun(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        np.random.seed(NUMPY_SEED)
        ndist = np.random.randn(*size)
        expected = np.sqrt(1 / size[0]) * ndist

        init = lecun_normal(*size, seed=NUMPY_SEED)
        assert np.array_equal(session.run(init), expected)


class TestUniformDistribution:

    def test_random_uniform(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        np.random.seed(NUMPY_SEED)
        udist = np.random.uniform(-1, 1, size=size)

        init = random_uniform(*size, seed=NUMPY_SEED)
        assert np.array_equal(session.run(init), udist)

    def test_xavier(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        np.random.seed(NUMPY_SEED)
        bound = np.sqrt(6/np.sum(size))
        udist = np.random.uniform(-bound, bound, size=size)

        init = xavier_uniform(*size, seed=NUMPY_SEED)
        assert np.array_equal(session.run(init), udist)

    def test_he(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        np.random.seed(NUMPY_SEED)
        bound = np.sqrt(6 / size[0])
        udist = np.random.uniform(-bound, bound, size=size)

        init = he_uniform(*size, seed=NUMPY_SEED)
        assert np.array_equal(session.run(init), udist)

    def test_lecun(self, session, test_case_unary):
        x = np.atleast_2d(test_case_unary)
        size = x.shape

        np.random.seed(NUMPY_SEED)
        bound = np.sqrt(3 / size[0])
        udist = np.random.uniform(-bound, bound, size=size)

        init = lecun_uniform(*size, seed=NUMPY_SEED)
        assert np.array_equal(session.run(init), udist)
