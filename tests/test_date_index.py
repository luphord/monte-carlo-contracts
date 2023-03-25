import unittest
import numpy as np
from mcc import DateIndex, At
from .test_utils import make_model


class TestDateIndex(unittest.TestCase):
    def test_date_index_specific_example(self) -> None:
        x = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.1, 1.1, 2.1, 3.1],
                [0.2, 1.2, 2.2, 3.2],
                [0.9, 0.8, 0.7, 0.6],
                [0.0, -1, -2, -3],
            ]
        )
        idx = DateIndex(np.array([1, 1, 2, 0, 3]))
        self.assertTrue(np.allclose([1.0, 1.1, 2.2, 0.9, -3], idx.index_column(x)))
        self.assertTrue(np.allclose([2, 1, 2, -1, -1], idx.next_after(x > 1).index))
        self.assertTrue(
            np.all(
                np.array(
                    [
                        [False, True, True, True],
                        [False, True, True, True],
                        [False, False, True, True],
                        [True, True, True, True],
                        [False, False, False, True],
                    ]
                )
                == idx.after_mask(4),
            )
        )
        self.assertTrue(
            np.all(
                np.array(
                    [
                        [False, False, True, True],
                        [False, True, True, True],
                        [False, False, True, True],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                )
                == idx.next_after(x > 1).after_mask(4),
            )
        )
        self.assertTrue(
            np.all(
                idx.next_after(x > 1).after_mask(10)
                == ~idx.next_after(x > 1).before_mask(10),
            )
        )

    def test_date_index(self) -> None:
        model = make_model()
        at0 = At(model.dategrid[0])
        idx0 = model.eval_date_index.next_after(
            at0.simulate(model.eval_date_index, model)
        )
        self.assertTrue((idx0.index == 0).all())
        at1 = At(model.dategrid[1])
        idx1 = model.eval_date_index.next_after(
            at1.simulate(model.eval_date_index, model)
        )
        self.assertTrue((idx1.index == 1).all())
