//! What a replay of a recorded session does differently from the session.
//!
//! A replay (`dehydration_hydration --replay <session file>`, started by the
//! Session Replay tool) feeds the window the input that was recorded, so it
//! does everything the user did — including running the correction and
//! exporting its result. None of that may land on the user's files, so in a
//! replay:
//!
//! * what a dialog chose to write (the export folder, a mask, the profiles,
//!   the settings) is written under the scratch folder of the replay
//!   (`/tmp/session_replay_<user>/<pid>/<its own path>`) — the session
//!   library does that for the answers of the save dialogs;
//! * so is the folder a calling application gave (`--called-from-app`,
//!   [`output`]);
//! * the tutorial window, a second process, is not started
//!   ([`starts_programs`]).

use std::path::PathBuf;

/// The folder to write to for `path`: itself, or in a replay its stand-in
/// under the scratch folder (created: the session's existed).
pub fn output(path: PathBuf) -> PathBuf {
    if !session_replay::is_replaying() {
        return path;
    }
    let to = session_replay::redirect_write(&path);
    let _ = std::fs::create_dir_all(&to);
    to
}

/// The tool starts the other programs it is asked for (the tutorial
/// window): not in a replay.
pub fn starts_programs() -> bool {
    !session_replay::is_replaying()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outside_a_replay_nothing_changes() {
        let path = PathBuf::from("/SNS/VENUS/IPTS-0/shared/corrected");
        assert_eq!(output(path.clone()), path);
        assert!(starts_programs());
    }
}
