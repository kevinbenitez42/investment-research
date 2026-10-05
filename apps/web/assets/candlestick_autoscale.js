/* Keep the price panel fitted to visible candles and enabled overlays. */
(function () {
    "use strict";

    function priceRange(traces, dateRange, logarithmic) {
        const bounds = dateRange.map(value => new Date(value).getTime()).sort((a, b) => a - b);
        let low = Infinity;
        let high = -Infinity;
        for (const trace of traces) {
            if (trace.visible === false || trace.visible === "legendonly") continue;
            if ((trace.yaxis || "y") !== "y" || (trace.xaxis || "x") !== "x") continue;
            const columns = trace.type === "candlestick" ? [trace.low, trace.high] : [trace.y];
            for (let i = 0; i < (trace.x || []).length; i++) {
                const date = new Date(trace.x[i]).getTime();
                if (!Number.isFinite(date) || date < bounds[0] || date > bounds[1]) continue;
                for (const column of columns) {
                    const raw = column && column[i];
                    if (raw == null) continue;
                    let value = Number(raw);
                    if (!Number.isFinite(value) || (logarithmic && value <= 0)) continue;
                    if (logarithmic) value = Math.log10(value);
                    low = Math.min(low, value);
                    high = Math.max(high, value);
                }
            }
        }
        if (!Number.isFinite(low)) return null;
        const padding = (high - low || Math.max(Math.abs(high), 1)) * 0.05;
        return [low - padding, high + padding];
    }

    // Also expose the calculation to the offline regression checks.
    if (typeof module !== "undefined" && module.exports) module.exports = {priceRange};
    if (typeof document === "undefined") return;

    const attached = new WeakSet();
    function attach(graph) {
        if (attached.has(graph) || typeof graph.on !== "function") return;
        attached.add(graph);
        let pending = false;
        function schedule() {
            if (pending) return;
            pending = true;
            requestAnimationFrame(() => {
                pending = false;
                if (!graph.isConnected || !window.Plotly || !graph._fullLayout) return;
                // Full traces contain Plotly's decoded numeric arrays.
                const traces = graph._fullData || [];
                if (!traces.some(t => t.type === "candlestick" && t.visible !== false)) return;
                const xaxis = graph._fullLayout.xaxis;
                const yaxis = graph._fullLayout.yaxis;
                if (!xaxis || !yaxis || !xaxis.range) return;
                const range = priceRange(traces, xaxis.range, yaxis.type === "log");
                if (!range) return; // Retain the previous scale for an empty date range.
                const current = yaxis.range || [];
                const tolerance = Math.max(1, Math.abs(range[0]), Math.abs(range[1])) * 1e-9;
                if (yaxis.autorange === false && current.length === 2 &&
                    current.every((value, i) => Math.abs(value - range[i]) < tolerance)) return;
                window.Plotly.relayout(graph, {"yaxis.range": range, "yaxis.autorange": false});
            });
        }
        graph.on("plotly_afterplot", schedule);
        graph.on("plotly_relayout", schedule);
        graph.on("plotly_restyle", schedule);
        schedule();
    }
    let scanPending = false;
    function scan() {
        if (scanPending) return;
        scanPending = true;
        requestAnimationFrame(() => {
            scanPending = false;
            document.querySelectorAll(".js-plotly-plot").forEach(attach);
        });
    }
    function start() {
        new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
        scan();
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
})();
