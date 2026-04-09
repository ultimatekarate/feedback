//! CouplingModel implementation for the 5-channel agentic pressure system.
//!
//! Laboratory layer: pure math. Encodes the coupling hypothesis —
//! which channels affect which, and with what coefficients.

use nalgebra::DMatrix;
use volterra_stability::config::{ImpulseRates, OperatingPoint, SystemConfig};
use volterra_stability::coupling::CouplingModel;
use volterra_stability::scaler::dscaler;

use crate::channels::{Channel, FeedbackConfig};

/// The agentic workflow coupling model.
///
/// Implements CouplingModel for the 5-channel pressure system.
/// The coupling coefficients encode the hypothesis about how agent
/// pressure channels interact. These are tunable parameters that
/// will be fitted from empirical session data.
pub struct AgentCouplingModel {
    config: FeedbackConfig,
}

impl AgentCouplingModel {
    pub fn new(config: FeedbackConfig) -> Self {
        Self { config }
    }
}

impl CouplingModel for AgentCouplingModel {
    fn config(&self) -> &SystemConfig {
        &self.config.system
    }

    fn build_jacobian(&self, rates: &ImpulseRates, op: &OperatingPoint) -> DMatrix<f64> {
        let dim = Channel::DIM;
        let mut j = DMatrix::zeros(dim, dim);

        let cfg = &self.config.system;
        let lambdas = cfg.lambdas();
        let criticals = cfg.criticals();

        let ctx = Channel::Context.index();
        let lat = Channel::Latency.index();
        let err = Channel::Error.index();
        let prg = Channel::Progress.index();
        let rep = Channel::Repetition.index();

        // Diagonal: self-decay minus self-coupling via scaler derivative.
        // J[i,i] = -lambda_i + u_i * d_sigma_i/dI_i
        for i in 0..dim {
            j[(i, i)] = -lambdas[i] + rates.rates[i] * dscaler(op.vals[i], criticals[i]);
        }

        // --- Cross-coupling (the hypothesis) ---

        // Cross-coupling coefficients are normalized by the critical
        // threshold of the target channel to keep the Jacobian entries
        // dimensionless and comparable. These are initial hypotheses —
        // all coefficients will be fitted from empirical session data.

        // Error -> Context: errors trigger read cascades that fill context.
        // Observed: 900x baseline read rate during errors.
        // J[ctx, err] > 0 (destabilizing)
        j[(ctx, err)] = 0.02 * lambdas[ctx];

        // Error -> Progress: errors stall forward progress.
        // J[prg, err] > 0 (destabilizing)
        j[(prg, err)] = 0.01 * lambdas[prg];

        // Error -> Repetition: errors cause re-reading the same files.
        // J[rep, err] > 0 (destabilizing)
        j[(rep, err)] = 0.02 * lambdas[rep];

        // Repetition -> Context: re-reading files fills context.
        // J[ctx, rep] > 0 (destabilizing)
        j[(ctx, rep)] = 0.01 * lambdas[ctx];

        // Context -> Error: the critical coupling. Does high context
        // pressure degrade the agent's ability to fix errors?
        // If positive (destabilizing), this creates the death spiral.
        // Conservative initial estimate — needs empirical validation.
        j[(err, ctx)] = 0.005 * lambdas[err];

        // Latency -> Repetition: high latency may cause retries.
        // Weak coupling, needs validation.
        j[(rep, lat)] = 0.005 * lambdas[rep];

        j
    }

    fn normalization_scales(&self) -> Vec<f64> {
        self.config.system.criticals()
    }

    fn lyapunov_matrix(&self) -> Option<DMatrix<f64>> {
        // Deferred — no Lyapunov certificate yet.
        // Will be computed once empirical coupling coefficients are fitted.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_stability::config::{ImpulseRates, OperatingPoint};

    fn test_config() -> FeedbackConfig {
        FeedbackConfig::from_defaults()
    }

    fn moderate_rates() -> ImpulseRates {
        // Moderate agentic workload: measured impulses from Phase 0 JONLs.
        // Rates are in channel-units per second at ~1 call every 8s.
        ImpulseRates::from_slice(&[
            50.0,  // context: ~400 tokens/call / 8s
            0.38,  // latency: ~3s per Bash, ~1 Bash per 8 calls
            0.01,  // error: ~2 errors per 300s session
            0.1,   // progress: 0.1/s base stall rate
            0.12,  // repetition: ~1.0 per call / 8s
        ])
    }

    #[test]
    fn jacobian_has_correct_dimension() {
        let model = AgentCouplingModel::new(test_config());
        let j = model.build_jacobian(&moderate_rates(), &OperatingPoint::idle(Channel::DIM));
        assert_eq!(j.nrows(), Channel::DIM);
        assert_eq!(j.ncols(), Channel::DIM);
    }

    #[test]
    fn diagonal_entries_are_negative_at_idle() {
        let model = AgentCouplingModel::new(test_config());
        let j = model.build_jacobian(&moderate_rates(), &OperatingPoint::idle(Channel::DIM));
        // At idle (all integrals at zero), diagonal should be -lambda + u*(-1/crit).
        // The scaler derivative at zero is -1/crit, and u*(-1/crit) is negative,
        // so diagonal = -lambda + negative = strongly negative.
        for i in 0..Channel::DIM {
            assert!(
                j[(i, i)] < 0.0,
                "diagonal [{i},{i}] should be negative at idle, got {}",
                j[(i, i)]
            );
        }
    }

    #[test]
    fn error_to_context_coupling_is_positive() {
        let model = AgentCouplingModel::new(test_config());
        let j = model.build_jacobian(&moderate_rates(), &OperatingPoint::idle(Channel::DIM));
        let ctx = Channel::Context.index();
        let err = Channel::Error.index();
        assert!(
            j[(ctx, err)] > 0.0,
            "error->context coupling should be positive (destabilizing)"
        );
    }
}
