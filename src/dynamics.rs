//! NonlinearDynamics implementation for the 6-channel agentic pressure system.
//!
//! Laboratory layer: pure math. Implements the full nonlinear right-hand side
//! for offline simulation and Lyapunov exponent computation.

use nalgebra::DVector;
use volterra_stability::config::ImpulseRates;
use volterra_stability::integrators::NonlinearDynamics;
use volterra_stability::scaler::linear_scaler;

use crate::channels::{Channel, FeedbackConfig};

/// Full nonlinear dynamics of the 6-channel agentic pressure system.
///
/// Unlike the linearized Jacobian (valid near one operating point), this
/// captures the full saturation nonlinearities across the entire state space.
pub struct AgentDynamics {
    config: FeedbackConfig,
    rates: ImpulseRates,
    /// Whether the agent is in a degraded mode (e.g., near context limit).
    perturbation_active: bool,
}

impl AgentDynamics {
    pub fn new(config: FeedbackConfig, rates: ImpulseRates) -> Self {
        Self {
            config,
            rates,
            perturbation_active: false,
        }
    }

    /// Compute the effective impulse rate for each channel given the current state.
    fn impulse_rates(&self, x: &DVector<f64>) -> Vec<f64> {
        let criticals = self.config.system.criticals();

        let sigma_ctx = linear_scaler(x[Channel::Context.index()], criticals[Channel::Context.index()]);
        let sigma_err = linear_scaler(x[Channel::Error.index()], criticals[Channel::Error.index()]);

        let ctx = Channel::Context.index();
        let cst = Channel::Cost.index();
        let lat = Channel::Latency.index();
        let err = Channel::Error.index();
        let prg = Channel::Progress.index();
        let rep = Channel::Repetition.index();

        // Error stress drives read cascades, repetition, and stalls.
        let error_stress = (x[err] / criticals[err]).min(1.0);

        vec![
            // Context: base rate + error-driven reads + repetition-driven re-reads
            self.rates.rates[ctx] * sigma_ctx
                + self.rates.rates[ctx] * 0.5 * error_stress
                + self.rates.rates[ctx] * 0.3 * (x[rep] / criticals[rep]).min(1.0),
            // Cost: always accumulating proportional to context activity
            self.rates.rates[cst] + self.rates.rates[cst] * 0.1 * (x[ctx] / criticals[ctx]).min(1.0),
            // Latency: base rate, weakly coupled
            self.rates.rates[lat],
            // Error: base rate + context-degradation coupling
            self.rates.rates[err] * sigma_err
                + self.rates.rates[err] * 0.1 * (x[ctx] / criticals[ctx]).min(1.0),
            // Progress: stall pressure grows when errors are active
            self.rates.rates[prg] * (1.0 + 0.3 * error_stress),
            // Repetition: base rate + error-driven re-reads
            self.rates.rates[rep] + self.rates.rates[rep] * 0.4 * error_stress,
        ]
    }
}

impl NonlinearDynamics for AgentDynamics {
    fn dim(&self) -> usize {
        Channel::DIM
    }

    fn rhs(&self, _t: f64, x: &DVector<f64>) -> DVector<f64> {
        let lambdas = self.config.system.lambdas();
        let f = self.impulse_rates(x);

        let mut dx = DVector::zeros(Channel::DIM);
        for i in 0..Channel::DIM {
            dx[i] = f[i] - lambdas[i] * x[i];
        }
        dx
    }

    fn set_perturbation(&mut self, active: bool) {
        self.perturbation_active = active;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_stability::integrators::rk4_step;

    fn test_dynamics() -> AgentDynamics {
        let config = FeedbackConfig::from_defaults();
        // Moderate agentic workload: measured impulses from Phase 0 JONLs.
        // Rates are in channel-units per second at ~1 call every 8s.
        // Measured token counts: Read=1175, Edit=200, Bash=565 tokens.
        // Typical mix: ~3 Edit + 1 Read + 0.5 Bash per 8 calls → avg ~400 tok/call.
        let rates = ImpulseRates::from_slice(&[
            50.0,  // context: ~400 tokens/call / 8s
            0.001, // cost: 0.005 per call / 8s + overhead
            0.38,  // latency: ~3s per Bash, ~1 Bash per 8 calls
            0.01,  // error: ~2 errors per 300s session
            0.1,   // progress: 0.1/s base stall rate
            0.12,  // repetition: ~1.0 per call / 8s
        ]);
        AgentDynamics::new(config, rates)
    }

    #[test]
    fn rhs_at_zero_is_positive() {
        let sys = test_dynamics();
        let x = DVector::zeros(Channel::DIM);
        let dx = sys.rhs(0.0, &x);
        // At zero state, impulse rates are positive and decay is zero,
        // so dx should be positive (pressure builds from rest).
        for i in 0..Channel::DIM {
            assert!(
                dx[i] >= 0.0,
                "rhs[{i}] at zero should be non-negative, got {}",
                dx[i]
            );
        }
    }

    #[test]
    fn system_reaches_equilibrium() {
        let sys = test_dynamics();
        let mut x = DVector::zeros(Channel::DIM);
        let dt = 1.0;
        // Evolve for a long time to approach equilibrium.
        // Cost channel has lambda=0.00003 (half-life ~6.4 hours), so
        // equilibrium requires simulating several half-lives.
        // 100,000 steps at dt=1.0 = 100,000 simulated seconds ≈ 27 hours.
        for _ in 0..100_000 {
            x = rk4_step(&sys, 0.0, &x, dt);
        }
        // At equilibrium, rhs should be near zero
        let dx = sys.rhs(0.0, &x);
        let residual = dx.norm();
        assert!(
            residual < 1.0,
            "system should approach equilibrium, residual norm = {}",
            residual
        );
    }
}
