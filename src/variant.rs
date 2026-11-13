//! Which build this is: production (`rust_dehydration_hydration`, branch
//! `main`) or the beta (`rust_dehydration_hydration_development`). Both
//! checkouts hold the same sources, so the variant comes from the name of
//! the checkout the binary was built in, not from a per-branch constant
//! that a merge would carry across.

/// The folder the crate was built in.
const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// True for a binary built in a `…_development` (or `…_beta`) checkout.
pub fn is_beta() -> bool {
    is_beta_dir(MANIFEST_DIR)
}

fn is_beta_dir(dir: &str) -> bool {
    let name = dir.trim_end_matches('/').rsplit('/').next().unwrap_or(dir);
    name.ends_with("_development") || name.ends_with("_beta")
}

/// `"mbirtorch hsnt"` or `"beta — mbirtorch hsnt"`: the parenthesis of the
/// window title, the About heading and the provenance file.
pub fn label() -> &'static str {
    if is_beta() { "beta — mbirtorch hsnt" } else { "mbirtorch hsnt" }
}

/// The stem of the shared log file: `dehydration_hydration` or
/// `dehydration_hydration_beta` (the beta logs to its own file).
pub fn log_stem() -> &'static str {
    if is_beta() { "dehydration_hydration_beta" } else { "dehydration_hydration" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_follows_the_checkout_name() {
        assert!(is_beta_dir("/SNS/VENUS/shared/software/git/rust_dehydration_hydration_development"));
        assert!(is_beta_dir("/x/rust_dehydration_hydration_development/"));
        assert!(is_beta_dir("/x/rust_dehydration_hydration_beta"));
        assert!(!is_beta_dir("/SNS/VENUS/shared/software/git/rust_dehydration_hydration"));
        assert!(!is_beta_dir("/x_development/rust_dehydration_hydration"));
    }
}
