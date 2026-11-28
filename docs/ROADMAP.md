# Roadmap

- Phase 0A: Native foundation and initial Pose Lab.
- Phase 0A.1: Foundation hardening, deterministic evidence, and TITAN qualification preparation.
- Phase 0A.2: TITAN dependency, build, camera-capability, and strict-CUDA investigation.
- Phase 0A.3: Audited CUDA execution, provider enforcement, and qualification recovery.
- Phase 0A.4: Live Pose Lab restoration, sustained-latency investigation, regression hardening, and final TITAN physical acceptance.
- Phase 0B: Comparative pose-model evaluation.
  - 0B.1: Immutable workload manifests, deterministic runs and comparable evidence.
  - 0B.2: RTMW evaluation.
  - 0B.3: MediaPipe Pose evaluation.
  - 0B.4: NVIDIA Maxine feasibility and evaluation, subject to SDK licensing and availability.
  - 0B.5: Production backend selection.
  - 0B.6: Backend registry, capability negotiation, game-specific profiles and safe model lifecycle.
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

Phase 0B.1 proceeds following the user's qualitative live Pose Lab smoke test and the near-60-FPS idle simulated performance run. This is an explicit scope decision, not completion of Phase 0A.4 physical qualification. Quantitative physical accuracy, actual camera delivery, complete provider/resource evidence review, and production memory isolation remain open. Representative recordings, annotated evaluation and physical qualification must be completed before backend selection can be called qualified. See [POSE-COMPARISON.md](POSE-COMPARISON.md).
