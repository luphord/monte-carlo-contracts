import unittest
from mcc import Model, One
from .test_utils import make_model


class TestModel(unittest.TestCase):
    def test_model_without_fx(self) -> None:
        model = make_model()
        model2 = Model(
            model.dategrid,
            {},
            {},
            model.simulated_stocks,
            model.numeraire,
            model.numeraire_currency,
        )
        indexedcf = One(model2.numeraire_currency).generate_cashflows(
            model2.eval_date_index, model
        )
        cf = model2.in_numeraire_currency(indexedcf)
        self.assertTrue((cf.cashflows["value"][:, 0] == 1).all())
        self.assertEqual(model2.currencies, {model2.numeraire_currency})

    def test_model_creation(self) -> None:
        nsim = 100
        model = make_model(nsim=nsim)
        self.assertEqual(model.shape, (nsim, model.dategrid.size))
