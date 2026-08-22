# Massive adaptive-alpha source hypotheses V1

The model may learn six qualified sources. These names are hypotheses, not
performance claims.

| Source | Intended information | Expected horizon profile |
|---|---|---|
| `slow_trend` | residual continuation conditioned on volume and volatility | 5–63 sessions |
| `fast_reversal` | short pressure, gaps, and temporary price displacement | 1–5 sessions |
| `intraday_path` | ordered opening, intraday, and closing price/volume path | 1–5 sessions |
| `tape_flow` | quote-free signed-flow proxies and persistent trade activity | 1–21 sessions |
| `liquidity_state` | illiquidity, venue concentration, latency, and inventory compensation | 5–63 sessions |
| `regime_interaction` | state-dependent interactions among frozen expert outputs | all buckets |

The tape does not reveal aggressor side or investor identity. Every inferred
direction field must be named `quote_free_signed_flow_proxy`. Historical quotes,
NBBO, depth, queue position, true spread, options, and fundamental-ratio alpha
are outside the Developer V1 claim.

A source earns the word alpha only when source-only, add-one, and leave-one-out
contrasts agree out of sample; signal-only net return and factor alpha have
positive lower bounds; costs survive; and neither one year nor one sector
dominates. Attention, saliency, or feature importance is not alpha evidence.
