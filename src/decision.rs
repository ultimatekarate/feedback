//! Governor decision types.
//!
//! Dictionary layer: inert nouns. What the governor can decide,
//! and the pressure snapshot that informed the decision.

use serde::{Deserialize, Serialize};

/// A snapshot of the current pressure state across all channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureSnapshot {
    /// Current integral values per channel.
    pub values: Vec<f64>,
    /// Pressure values from the previous tool call, if available.
    /// Used by the monitor to compute trajectory (rising/falling/stable)
    /// and include it in deny/warn messages so the agent can distinguish
    /// transient pressure (attrition is viable) from escalating pressure
    /// (strategy change is needed).
    #[serde(default)]
    pub prev_values: Option<Vec<f64>>,
    /// Composite stress (weighted sum of scaler values).
    pub composite_stress: f64,
    /// Spectral gap gamma_1. Distance from instability boundary.
    pub spectral_gap: f64,
    /// Whether all eigenvalues have negative real parts.
    pub is_stable: bool,
}

/// What the governor can do to a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decision {
    /// Allow the tool call to proceed.
    Allow,
    /// Block the tool call with a reason the agent will see.
    Deny { reason: String },
    /// Allow but modify the tool call (e.g., limit scope).
    Modify { reason: String, patch: ToolPatch },
}

/// Modifications the governor can apply to a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPatch {
    /// If set, override the tool's input arguments.
    pub updated_input: Option<serde_json::Value>,
    /// Additional context injected into the agent's prompt.
    pub additional_context: Option<String>,
}

/// A governor verdict: the decision plus the state that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorVerdict {
    /// The decision made.
    pub decision: Decision,
    /// The pressure snapshot at the time of the decision.
    pub snapshot: PressureSnapshot,
    /// Wall-clock timestamp (seconds since session start).
    pub timestamp: f64,
    /// Which tool call triggered this verdict.
    pub tool_name: String,
}
