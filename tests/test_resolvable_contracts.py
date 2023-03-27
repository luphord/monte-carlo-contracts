import unittest
import numpy as np
from mcc import (
    ResolvableContract,
    One,
    ZeroCouponBond,
    EuropeanOption,
    generate_cashflows,
)
from .test_utils import make_model, MyContract


class TestResolvableContracts(unittest.TestCase):
    def test_resolvable_contract_creation(self) -> None:
        model = make_model()
        c = MyContract(model.dategrid[-1], 1234)
        generate_cashflows(model, c.resolve())
        self.assertRaises(TypeError, lambda: ResolvableContract())  # type: ignore

    def test_zero_coupon_bond(self) -> None:
        model = make_model()
        notional = 1234
        currency = "USD"
        zcb = ZeroCouponBond(model.dategrid[-2], notional, currency)
        cf = generate_cashflows(model, zcb.resolve())
        self.assertEqual(cf.currencies.shape, (1,))
        self.assertEqual(cf.currencies[0], currency)
        self.assertEqual(cf.cashflows.shape, (model.nsim, 1))
        self.assertTrue((cf.cashflows["value"] == notional).all())
        self.assertTrue((cf.cashflows["date"] == model.dategrid[-2]).all())

    def test_european_option_on_zcb(self) -> None:
        model = make_model()
        notional = 1234
        currency = "USD"
        strike = 1000
        zcb = ZeroCouponBond(model.dategrid[-2], notional, currency)
        opt = EuropeanOption(model.dategrid[-2], zcb.resolve() - strike * One(currency))
        cf = generate_cashflows(model, opt.resolve())
        self.assertEqual(cf.currencies.shape, (3,))
        self.assertEqual(cf.currencies[0], currency)
        self.assertEqual(cf.currencies[1], currency)
        self.assertEqual(cf.currencies[2], "NNN")
        self.assertEqual(cf.cashflows.shape, (model.nsim, 3))
        self.assertTrue((cf.cashflows["date"][:, 0] == model.dategrid[-2]).all())
        self.assertTrue((cf.cashflows["value"][:, 0] == notional).all())
        self.assertTrue((cf.cashflows["date"][:, 1] == model.dategrid[-2]).all())
        self.assertTrue((cf.cashflows["value"][:, 1] == -strike).all())
        self.assertTrue((np.isnat(cf.cashflows["date"][:, 2])).all())
        self.assertTrue((cf.cashflows["value"][:, 2] == 0).all())
