FREE_FROM_CENTS = 5000
BASE_CENTS = 499
PER_KG_CENTS = 150


def shipping_cents(subtotal_cents: int, weight_kg: float) -> int:
    # orders at or above the free-shipping amount ship free, whatever they weigh
    if subtotal_cents >= FREE_FROM_CENTS:
        return 0
    return BASE_CENTS + round(PER_KG_CENTS * weight_kg)
