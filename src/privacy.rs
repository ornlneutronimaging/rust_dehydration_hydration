//! Private mode: the user's choice not to have their sessions recorded.
//!
//! By default the Dehydration / Hydration Correction tool records what is done in its window (see the session
//! log in `main.rs`), so that a problem can be reproduced step by step by
//! the imaging team with the Session Replay tool. A user who does not want
//! that turns the private mode on, with the 🔒 button of the toolbar: the
//! session being recorded is deleted, and the choice is remembered
//! (`venus_rust_tools/dehydration_hydration_private_mode`) for the next starts, until it
//! is turned off again. The recordings of their earlier sessions (kept 30
//! days) are deleted too if they ask for it.
//!
//! Nobody is recorded without having been told: the first time the tool
//! records a session of a user, a notice says what is recorded, where, who
//! can read it, for how long, and how to turn it off ([`notice_begin`]).
//!
//! The same module as the one of the Neutron Image Viewer, with the names
//! and the texts of the Dehydration / Hydration Correction tool.

use crate::logger;
use std::path::PathBuf;

/// The name the tool records its sessions under.
pub const APP: &str = "dehydration_hydration";

const WARN: egui::Color32 = egui::Color32::from_rgb(230, 170, 60);

/// The preferences of the VENUS rust tools, under `$XDG_CONFIG_HOME` (or
/// `~/.config`).
fn prefs_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).filter(|p| !p.as_os_str().is_empty()).or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("venus_rust_tools"))
}

/// The preference file: there when the private mode is on.
fn pref_path() -> Option<PathBuf> {
    Some(prefs_dir()?.join("dehydration_hydration_private_mode"))
}

/// The preference file of the recording notice: missing until the tool
/// first records a session of this user, then `pending` while the notice is
/// to be shown and `seen` once the user answered it. One of the preference
/// files saved with a session (`main.rs`), so that a replay shows the notice
/// exactly when the recorded session did — the click that closed it is part
/// of the recording.
pub const NOTICE_FILE: &str = "dehydration_hydration_recording_notice";
const PENDING: &str = "pending";
const SEEN: &str = "seen";

fn notice_state() -> Option<String> {
    std::fs::read_to_string(prefs_dir()?.join(NOTICE_FILE)).ok().map(|s| s.trim().to_owned())
}

fn set_notice_state(state: &str) -> bool {
    let Some(dir) = prefs_dir() else { return false };
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(dir.join(NOTICE_FILE), format!("{state}\n")).is_ok()
}

/// A session of this user is about to be recorded: is the notice to be
/// shown? Yes until they answered it once. Call before the recording starts
/// (it marks the notice as pending, which the session has to save).
pub fn notice_begin() -> bool {
    if notice_state().as_deref() == Some(SEEN) {
        return false;
    }
    set_notice_state(PENDING);
    true
}

/// The notice was pending when the session being replayed started (read
/// from the preferences of that session).
pub fn notice_pending() -> bool {
    notice_state().as_deref() == Some(PENDING)
}

/// The user asked not to be recorded.
pub fn load() -> bool {
    pref_path().is_some_and(|p| p.is_file())
}

/// Remember the choice. Returns false when it could not be saved (the
/// private mode then only holds for this session).
pub fn save(on: bool) -> bool {
    let Some(path) = pref_path() else { return false };
    if on {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&path, "on\n").is_ok()
    } else {
        std::fs::remove_file(&path).is_ok() || !path.exists()
    }
}

/// What the window is about.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Asking {
    /// The first session recorded: what is recorded and how to turn it off.
    Notice,
    /// Confirm turning the private mode on.
    TurnOn,
    /// The private mode was just turned off: when recording resumes.
    TurnedOff,
}

/// The private mode of this run of the tool: its button and its window.
pub struct Private {
    /// Sessions are not recorded.
    pub on: bool,
    asking: Option<Asking>,
    /// The user's earlier recordings, counted when the window opens, and
    /// their choice to have them deleted with the private mode turned on.
    past: usize,
    delete_past: bool,
}

impl Private {
    pub fn new(on: bool) -> Self {
        Self { on, asking: None, past: 0, delete_past: false }
    }

    /// Open the window on `asking`, with the number of recordings of the
    /// user's earlier sessions (the one in progress left out).
    fn ask(&mut self, asking: Asking) {
        self.delete_past = false;
        // A replay runs on somebody else's session: one's own recordings
        // are none of its business.
        self.past = if session_replay::is_replaying() {
            0
        } else {
            let all = session_replay::format::own_sessions(&session_replay::format::sessions_dir(), APP).len();
            all.saturating_sub(session_replay::is_recording() as usize)
        };
        self.asking = Some(asking);
    }

    /// The choice to delete the earlier recordings too. Always there, so
    /// that the buttons under it are where they were in a recorded session.
    fn past_checkbox(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.delete_past, format!("Also delete the recordings of my earlier sessions ({})", self.past))
            .on_hover_text(format!("The recordings of your previous sessions in this tool. Left alone, each is deleted {} days after its session.", session_replay::format::RETENTION_DAYS));
    }

    /// The button of the toolbar: lit while the private mode is on.
    pub fn toolbar_button(&mut self, ui: &mut egui::Ui) {
        let (text, tip) = if self.on {
            (egui::RichText::new("🔒 Private").color(WARN), "Private mode is on: this session is not recorded — if something goes wrong, the imaging team cannot replay what you did. Click to turn it off.")
        } else {
            (egui::RichText::new("🔒"), "Private mode: do not record my sessions. By default what is done in this window is recorded, so that the imaging team can replay a session step by step and see a problem as you saw it.")
        };
        if ui.button(text).on_hover_text(tip).clicked() {
            if self.on { self.turn_off() } else { self.ask(Asking::TurnOn) }
        }
    }

    /// Show the recording notice (see [`notice_begin`]).
    pub fn notify(&mut self) {
        logger::log("Recording notice shown at start-up");
        self.ask(Asking::Notice);
    }

    fn turn_on(&mut self) {
        self.on = true;
        let deleted = session_replay::stop_recording(true);
        let saved = save(true);
        // Never in a replay: the recordings there would be the ones of
        // whoever replays, not of the user of the session.
        let past = (self.delete_past && !session_replay::is_replaying()).then(|| session_replay::format::delete_own_sessions(&session_replay::format::sessions_dir(), APP));
        // Written after the recording stopped: to the application log only.
        logger::log(format!(
            "Private mode turned on: session recording stopped{}{}{}",
            if deleted.is_some() { ", this session's file deleted" } else { "" },
            past.map(|n| format!(", {n} of {} earlier recording(s) deleted", self.past)).unwrap_or_default(),
            if saved { "" } else { " (the choice could not be saved for the next starts)" }
        ));
    }

    fn turn_off(&mut self) {
        self.on = false;
        save(false);
        logger::log("Private mode turned off: sessions are recorded again from the next start of the tool");
        self.asking = Some(Asking::TurnedOff);
    }

    /// The confirmation / information window.
    pub fn window(&mut self, ctx: &egui::Context) {
        let Some(asking) = self.asking else { return };
        let mut close = false;
        let title = if asking == Asking::Notice { "Your sessions are recorded" } else { "Private mode" };
        egui::Window::new(title).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.set_max_width(480.0);
            match asking {
                Asking::Notice => {
                    ui.label("This tool records what is done in its window — the data selected, the pointer, the clicks, the keys, what is typed or pasted, the answers of the file dialogs — so that, when something goes wrong, the imaging team can replay your session step by step and see the problem as you saw it. The data itself is not copied.");
                    ui.add_space(6.0);
                    ui.label(format!(
                        "The recordings are files in {}, named after your user name. They can be read by you and by the imaging team only, and are deleted after {} days.",
                        session_replay::format::sessions_dir().display(),
                        session_replay::format::RETENTION_DAYS
                    ));
                    ui.add_space(6.0);
                    ui.label("You can turn the recording off at any time with the 🔒 button at the top right of the window (private mode), which also deletes the recording of the session in progress. This notice is shown once.");
                    ui.add_space(6.0);
                    self.past_checkbox(ui);
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("OK, keep recording").clicked() {
                            set_notice_state(SEEN);
                            logger::log("Recording notice: recording kept");
                            close = true;
                        }
                        if ui.button("Turn private mode on").clicked() {
                            set_notice_state(SEEN);
                            self.turn_on();
                            close = true;
                        }
                    });
                }
                Asking::TurnOn => {
                    ui.label("This tool records what is done in its window — the data selected, the pointer, the clicks, the keys — so that, when something goes wrong, the imaging team can replay your session step by step and see the problem as you saw it.");
                    ui.add_space(6.0);
                    ui.colored_label(WARN, "In private mode your sessions are not recorded: helping you will not be as easy, since the imaging team will have to rely on your description of what happened.");
                    ui.add_space(6.0);
                    ui.label("Turning it on deletes the recording of this session and stays in effect the next times you start this tool, until you turn it off (the same 🔒 button).");
                    ui.add_space(6.0);
                    self.past_checkbox(ui);
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Keep recording (recommended)").clicked() {
                            close = true;
                        }
                        if ui.button("Turn private mode on").clicked() {
                            self.turn_on();
                            close = true;
                        }
                    });
                }
                Asking::TurnedOff => {
                    ui.label("Private mode is off. A session can only be replayed from its start, so this one stays unrecorded: recording resumes the next time you start this tool.");
                    ui.add_space(8.0);
                    if ui.button("OK").clicked() {
                        close = true;
                    }
                }
            }
        });
        if close {
            self.asking = None;
        }
    }
}
