import { discountCents, isValid } from "./coupons.js";

export function subtotalCents(items) {
  return items.reduce((sum, item) => sum + item.priceCents * item.qty, 0);
}

export function total(items, coupon, today) {
  const sum = subtotalCents(items);
  if (!coupon || !isValid(coupon, today)) return sum;
  return sum - discountCents(sum, coupon);
}
