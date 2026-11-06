// A coupon is { code, percentOff, expires }, where expires is a date like "2026-04-30".
// Dates in that form sort as strings, so comparing them as strings compares the days.

export function isValid(coupon, today) {
  return today <= coupon.expires;
}

export function discountCents(subtotalCents, coupon) {
  return Math.round((subtotalCents * coupon.percentOff) / 100);
}
