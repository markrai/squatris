# Roadmap

- Phase 0A: Native foundation and initial Pose Lab.
- Phase 0A.1: Foundation hardening, deterministic evidence, and TITAN qualification preparation.
- Phase 0B: Comparative pose-model evaluation.
- Phase 0C: Recorded-session infrastructure and physical pose qualification.
- Phase 1A: Player calibration and temporal pose filtering.
- Phase 1B: Player identity and tracking-loss handling.
- Phase 2A: Squat recognition.
- Phase 2B: Lateral movement recognition.
- Phase 2C: Independent arm-swipe recognition.
- Phase 2D: Gesture arbitration and combined-motion qualification.
- Phase 3A: Engine-independent falling-block gameplay.
- Phase 3B: Deterministic gameplay replay and input integration.
- Phase 4A: Godot GDExtension integration.
- Phase 4B: Playable Squatris.
- Phase 5: Graphics, audio and visual polish.
- Phase 6: Extended physical qualification and release readiness.
- Phase 7: Reusable motion-input SDK extraction, if justified by proven implementations.

Future gestures will use temporal state machines, player calibration, hysteresis, confidence gating, and explicit rearming. Tracking loss will fail closed. Qualification will report intentional recall and unintended activations separately.

Phase 0B must not start until Phase 0A.1 is reviewed and the hardened pipeline is qualified on TITAN. A ready-for-review status on VENOM does not satisfy the TITAN gate.
