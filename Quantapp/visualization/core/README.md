# Visualization Core

This folder is for shared visualization utilities that are reused across multiple views.

Examples of good fits:

- theme helpers
- shared layout controls
- axis or range helpers
- common figure-level utilities

Theme helpers in this folder are opt-in. Importing `Quantapp.visualization` or a view module should not apply notebook renderer settings or figure themes; notebooks/views should call the helper explicitly when they want that style.

Avoid putting one-off chart logic here just because it is technically reusable in theory.
