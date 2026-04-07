//! Feedback: agentic workflow stability governor.
//!
//! Uses Volterra integral pressure analysis to monitor AI agent sessions
//! in real time and intervene when pressure cascades threaten to destabilize
//! the workflow.
//!
//! # Architecture
//!
//! Three layers following the Linguistic Code Model:
//! - **Dictionary** (`channels`, `decision`, `hook_protocol`): inert nouns
//! - **Laboratory** (`pressure_model`, `dynamics`, `monitor`): pure math
//! - **Hands** (`governor`, `hooks`, `session`): IO-dependent runtime

// Dictionary layer — inert nouns
pub mod channels;
pub mod decision;
pub mod hook_protocol;

// Laboratory layer — pure math
pub mod pressure_model;
pub mod dynamics;
pub mod monitor;

// Hands layer — IO-dependent
pub mod governor;
pub mod hooks;
pub mod session;
