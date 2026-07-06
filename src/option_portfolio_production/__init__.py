"""Production-grade scaffolding for option-only Markowitz.

The package is separate from the research paper code on purpose.  Research ledgers can be
valid point-in-time evidence while still failing production verification because exact
settlement, fill, margin, assignment, broker, and vendor-reconciliation ledgers are absent.
"""
from .broker import BrokerAdapter, PaperBrokerAdapter
from .execution import FeeSchedule, OrderPolicy, build_execution_ledger, build_fill_ledger, estimate_nbbo_fill, target_weights_to_orders
from .lifecycle import AssignmentEvent, assign_short_option, early_exercise_risk, expiry_payoff
from .margin import MarginConfig, build_margin_ledger, conservative_order_margin, stress_loss_for_order
from .market_data import QuoteReconciliationResult, build_market_data_ledger, reconcile_quote_pair, validate_timestamp_monotonicity
from .optimizer import ProductionOptimizerConfig, post_cost_expected_returns
from .repair import REPAIR_LEDGER_COLUMNS, RepairEvent, RepairPolicy, adverse_drift_bps, attempt_order_repair, build_repair_ledger, propose_repair_order
from .risk import RiskGateConfig, evaluate_pre_trade_gate
from .schemas import AccountState, Fill, GateResult, MarginEstimate, OptionContract, OptionOrder, Position, QuoteSnapshot
from .settlement import attach_exact_vix_settlement, load_vro_soq_table, normalize_vro_soq_frame, require_exact_vix_settlement, settlement_coverage
from .shadow import SHADOW_FILL_MODEL, ShadowRunConfig, ShadowVerifier, load_shadow_quotes, load_shadow_targets, run_shadow_rebalance, write_shadow_outputs

__all__ = [
    "AccountState",
    "AssignmentEvent",
    "BrokerAdapter",
    "FeeSchedule",
    "Fill",
    "GateResult",
    "MarginConfig",
    "MarginEstimate",
    "OptionContract",
    "OptionOrder",
    "OrderPolicy",
    "PaperBrokerAdapter",
    "Position",
    "ProductionOptimizerConfig",
    "QuoteReconciliationResult",
    "QuoteSnapshot",
    "REPAIR_LEDGER_COLUMNS",
    "RepairEvent",
    "RepairPolicy",
    "RiskGateConfig",
    "SHADOW_FILL_MODEL",
    "ShadowRunConfig",
    "ShadowVerifier",
    "assign_short_option",
    "attach_exact_vix_settlement",
    "adverse_drift_bps",
    "attempt_order_repair",
    "build_execution_ledger",
    "build_fill_ledger",
    "build_margin_ledger",
    "build_market_data_ledger",
    "build_repair_ledger",
    "conservative_order_margin",
    "early_exercise_risk",
    "estimate_nbbo_fill",
    "evaluate_pre_trade_gate",
    "expiry_payoff",
    "load_vro_soq_table",
    "load_shadow_quotes",
    "load_shadow_targets",
    "normalize_vro_soq_frame",
    "post_cost_expected_returns",
    "propose_repair_order",
    "reconcile_quote_pair",
    "require_exact_vix_settlement",
    "run_shadow_rebalance",
    "settlement_coverage",
    "stress_loss_for_order",
    "target_weights_to_orders",
    "validate_timestamp_monotonicity",
    "write_shadow_outputs",
]
