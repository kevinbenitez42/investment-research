const assert = require("node:assert/strict");
const {priceRange} = require("../apps/web/assets/candlestick_autoscale.js");

const candles = {
    type: "candlestick",
    x: ["2025-01-01", "2025-01-02", "2025-01-03"],
    low: new Float64Array([1, 100, 105]),
    high: new Float64Array([1000, 110, 115]),
};
const zoom = ["2025-01-02", "2025-01-03"];
assert.deepEqual(priceRange([candles], zoom, false), [99.25, 115.75]);
assert.deepEqual(priceRange([candles], [...zoom].reverse(), false), [99.25, 115.75]);
const overlay = {type: "scatter", x: candles.x, y: [2000, 120, 125]};
assert.deepEqual(priceRange([candles, overlay], zoom, false), [98.75, 126.25]);
for (const visible of [false, "legendonly"]) {
    assert.deepEqual(priceRange([candles, {...overlay, visible}], zoom, false), [99.25, 115.75]);
}
assert.deepEqual(priceRange([candles, {...overlay, yaxis: "y2"}], zoom, false), [99.25, 115.75]);
assert.equal(priceRange([candles], ["2026-01-01", "2026-02-01"], false), null);
assert.deepEqual(priceRange([{type: "candlestick", x: [zoom[0]], low: [100], high: [100]}], zoom, false), [95, 105]);
assert.deepEqual(priceRange([{type: "scatter", x: zoom, y: [null, NaN]}], zoom, false), null);
const full = priceRange([candles], [candles.x[0], candles.x[2]], false);
assert.ok(full[0] < 1 && full[1] > 1000);
const log = priceRange([candles], zoom, true);
assert.ok(log[0] < Math.log10(100) && log[1] > Math.log10(115));
console.log("Candlestick autoscale regression checks passed.");
