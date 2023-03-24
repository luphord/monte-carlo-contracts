import unittest

import numpy as np
from pandas.testing import assert_frame_equal
from mcc import (
    IndexedCashflows,
    DateIndex,
    generate_cashflows,
    generate_simple_cashflows,
    generate_simple_cashflows_in_currency,
    generate_simple_cashflows_in_numeraire_currency,
    One,
    Cond,
)

from .test_utils import make_model, AlternatingBool


class TestCashflows(unittest.TestCase):
    def test_simple_cashflows(self) -> None:
        model = make_model()
        c = Cond(AlternatingBool(), One("EUR"), One("USD"))
        cf = generate_cashflows(model, c)
        simplecf = cf.to_simple_cashflows()
        self.assertEqual(simplecf.shape[1], model.nsim)
        self.assertEqual(simplecf.shape[0], 2)
        assert_frame_equal(simplecf, generate_simple_cashflows(model, c))
        simplecf2 = generate_simple_cashflows_in_currency(model, c, "USD")
        self.assertEqual(simplecf2.shape[1], model.nsim)
        self.assertEqual(simplecf2.shape[0], 1)
        self.assertEqual(simplecf2.index[0][1], "USD")
        simplecf3 = generate_simple_cashflows_in_numeraire_currency(model, c)
        self.assertEqual(simplecf3.shape[1], model.nsim)
        self.assertEqual(simplecf3.shape[0], 1)
        self.assertEqual(simplecf3.index[0][1], model.numeraire_currency)

    def test_cashflows(self) -> None:
        n = 10
        k = 2
        dategrid = np.array([np.datetime64("2030-07-14"), np.datetime64("2031-07-14")])
        cf1 = IndexedCashflows(
            np.zeros((n, k), dtype=IndexedCashflows.dtype),
            np.array(["USD"] * k, dtype=(np.unicode_, 3)),
            dategrid,
        )
        self.assertEqual(cf1.nsim, n)
        self.assertEqual(cf1.ncashflows, k)
        self.assertEqual(cf1.currencies.size, k)
        cf2 = IndexedCashflows(
            np.array(
                [
                    [(0, 123.45)],
                    [(1, 123.45)],
                ],
                IndexedCashflows.dtype,
            ),
            np.array(["USD"], dtype=(np.unicode_, 3)),
            dategrid,
        )
        self.assertEqual(cf2.nsim, 2)
        self.assertEqual(cf2.ncashflows, 1)
        self.assertEqual(cf2.currencies.size, 1)
        cf3 = cf1 + cf1
        self.assertTrue((cf3.cashflows[:, :k] == cf1.cashflows).all())
        self.assertTrue((cf3.cashflows[:, k:] == cf1.cashflows).all())
        cf3.apply_index()

    def test_cashflow_shifting(self) -> None:
        dategrid = np.array([np.datetime64("2030-07-14"), np.datetime64("2031-07-14")])
        cf = IndexedCashflows(
            np.array(
                [
                    [(0, 123.45)],
                    [(1, 123.45)],
                ],
                IndexedCashflows.dtype,
            ),
            np.array(["USD"], dtype=(np.unicode_, 3)),
            dategrid,
        )
        # shift to same index as cashflows are anyway
        shifted1 = cf.delay(DateIndex(np.array([0, 1])))
        self.assertEqual(cf, shifted1)
        # shift to index 0, meaning effectively no change
        shifted2 = cf.delay(DateIndex(np.array([0, 0])))
        self.assertEqual(cf, shifted2)
        # shift to index 1, meaning all cashflows at 1
        shifted3 = cf.delay(DateIndex(np.array([1, 1])))
        self.assertTrue((shifted3.cashflows["index"] == 1).all())
        # shift to index -1 (never), implying no shift
        shifted4 = cf.delay(DateIndex(np.array([-1, -1])))
        self.assertEqual(cf, shifted4)

    def test_negative_date_index(self) -> None:
        dategrid = np.array([np.datetime64("2030-07-14"), np.datetime64("2031-07-14")])
        cf = IndexedCashflows(
            np.array(
                [
                    [(0, 123.45)],
                    [(-1, 123.45)],
                ],
                IndexedCashflows.dtype,
            ),
            np.array(["USD"], dtype=(np.unicode_, 3)),
            dategrid,
        ).apply_index()
        self.assertEqual(cf.cashflows["date"][0], dategrid[0])
        self.assertTrue(np.isnat(cf.cashflows["date"][1]))
