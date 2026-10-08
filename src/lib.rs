//! Dehydration/Hydration correction library (beta, mbirtorch hsnt): stack
//! loading, the hand-off to the maximum-likelihood dehydrate/rehydrate
//! factorization of `mbirtorch.hsnt` (Harel Dor's `hsnt` branch, run as a
//! Python subprocess in its own pixi environment), and TIFF export.
//!
//! The GUI binary (`main.rs`) is a thin shell around these modules; they are
//! exposed here so they can be unit/integration tested without a display.

pub mod app;
pub mod colormap;
pub mod config;
pub mod correction;
pub mod export;
pub mod hsnt_cli;
pub mod loader;
pub mod mask;
pub mod recent;
pub mod run_lookup;
pub mod spectra;
pub mod theme;
pub mod zoom;
