# Testing

Normal tests require no camera, NVIDIA GPU, CUDA, or model file.

```powershell
ctest --preset windows-debug
```

The suite covers:

- Canonical coordinate normalization and bounds.
- COCO/HALPE mapping and anatomical left/right identity.
- Explicit missing landmarks, invalid confidence, and non-finite values.
- Joint angles, low-confidence inputs, and degenerate geometry.
- Timestamp ordering.
- Latest-frame replacement, stale-frame timeout, and close behavior.
- Synthetic acquisition, mock inference, disconnection propagation, and idempotent shutdown.
- Benchmark JSON smoke execution.

Hardware checks are opt-in, separately labeled, and excluded from normal CI. Configure them only on the target machine after acquiring the models and recording a qualification video:

```powershell
cmake --preset windows-cuda-release -DSQUATRIS_ENABLE_HARDWARE_TESTS=ON -DSQUATRIS_HARDWARE_VIDEO="$PWD\captures\qualification.mp4"
cmake --build --preset windows-cuda-release
ctest --preset windows-hardware
```

Live-camera behavior remains a visual/manual check:

```powershell
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --list-cameras
.\out\build\windows-cuda-release\apps\pose-lab\SquatrisPoseLab.exe --camera 0 --backend rtmpose --provider cuda
```

A passing mock test does not imply that cameras, model files, CUDA, or physical poses work. Qualification evidence belongs in a benchmark result and the checklist in `QUALIFICATION.md`.
