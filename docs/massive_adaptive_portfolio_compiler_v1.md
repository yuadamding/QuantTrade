# Massive adaptive portfolio compiler V1

The compiler consumes frozen bucketed alpha forecasts and uncertainty. It has
no position-age input and no duration preference.

For each security it forms cumulative alpha and calibrated uncertainty curves.
Prospective increases pay buy and expected exit costs; existing notional pays
only expected future exit cost. A trade occurs only when its expected
risk-adjusted improvement exceeds transaction cost, uncertainty, and
incremental portfolio risk.

The canonical constraints are long-only total weights with explicit CASH,
1% per security, 1.5% per issuer, 6% annualized tracking error, 0.10 absolute
active beta, 10% daily discretionary one-way turnover, and 2% trailing-ADV
participation. Corporate-action and terminal exits are forced and separately
attributed.

Only the first action from each receding-horizon solution is executed. The next
session recomputes forecasts and solves again. The maximizing forecast horizon
is diagnostic, not an exit date. A position persists only because no available
replacement has greater expected net value.

The future canonical solver is CPU float64 with create-only evidence for primal
and dual residuals, constraint violations, objective descent, KKT surrogate,
iteration count, convergence, and input/output receipts. The engineering RL
controller may adjust bounded alpha and risk preferences or tighten
discretionary turnover, but the deterministic compiler remains the sole
target-weight and feasibility authority. The controller has no temporal
release control; its zero action must reproduce the deterministic decision and
complete economic transition exactly.
