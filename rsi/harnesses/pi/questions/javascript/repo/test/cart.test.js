import assert from "node:assert/strict";
import { test } from "node:test";
import { subtotalCents, total } from "../src/cart.js";

const items = [
  { name: "mug", priceCents: 1200, qty: 2 },
  { name: "tea", priceCents: 650, qty: 1 },
];
const coupon = { code: "SPRING10", percentOff: 10, expires: "2026-04-30" };

test("the subtotal adds up every item", () => {
  assert.equal(subtotalCents(items), 3050);
});

test("a coupon works through its expiry date", () => {
  assert.equal(total(items, coupon, "2026-04-30"), 2745);
});

test("an expired coupon changes nothing", () => {
  assert.equal(total(items, coupon, "2026-05-01"), 3050);
});
