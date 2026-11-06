import unittest

from shipping import shipping_cents


class ShippingTest(unittest.TestCase):
    def test_an_order_of_the_free_shipping_amount_ships_free(self):
        self.assertEqual(shipping_cents(5000, 2.0), 0)

    def test_a_smaller_order_pays_the_base_and_its_weight(self):
        self.assertEqual(shipping_cents(4999, 2.0), 799)


if __name__ == "__main__":
    unittest.main()
