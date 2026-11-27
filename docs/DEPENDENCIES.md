# Dependencies

## Pinned matrix

| Component | Pin | Purpose |
|---|---:|---|
| C++ | C++20 | Production language |
| CMake | Minimum 3.28 | Configure and build |
| vcpkg registry | `9fa95bf687fa6ae7f2da0673a8067e5fc1515c27` | Manifest dependency resolution |
| ONNX Runtime | 1.20.1 | CPU and CUDA inference |
| CUDA | 12.x | Required by the selected ORT GPU package |
| cuDNN | 9.x | Required by the selected ORT GPU package |
| OpenCV | 4.14.0 | Capture, image transforms, diagnostics |
| GoogleTest | 1.18.0 | Tests |
| nlohmann/json | 3.12.0 | Benchmark output |
| RTMPose-m HALPE-26 | Manifest checksum | Initial pose model |
| YOLOX-tiny HumanArt | Manifest checksum | Person detector |

The vcpkg commit pins transitive port versions and build recipes. `vcpkg.json` disables unnecessary OpenCV default features and explicitly enables HighGUI, Media Foundation, and FFmpeg for diagnostics, live Windows capture, and prerecorded benchmarks.

## Compatibility rationale

ONNX Runtime's official CUDA provider matrix states that 1.20.x CUDA 12 packages use cuDNN 9.x; cuDNN major versions are not interchangeable. This older pinned runtime is intentional because later runtime releases changed minimum CUDA requirements. TITAN must be inventoried before any CUDA or cuDNN installation decision.

The acquisition script uses Microsoft NuGet packages because the official GitHub repository and release downloads returned 404 during the 2026-11-26 environment check. Package version, Microsoft package identity, embedded repository commit `5c1b7ccbff7e5141c1da7a9d963d660e5741c319`, and NuGet SHA-512 are pinned. CPU and GPU variants install into `.deps/onnxruntime/cpu` and `.deps/onnxruntime/gpu`; a managed-install marker prevents an existing directory or one variant from being overwritten by the other.

Bootstrap scripts fail on native command errors, reject filesystem roots and unexpected existing destinations, stage downloads in unique directories, verify content before placement, and are idempotent after a verified install. Model acquisition reads URLs, filenames, and both hashes exclusively from `models/manifest.json`.

## Security and licensing

- No runtime network request, account, telemetry service, or cloud inference is part of the application.
- Dependency downloads occur only in explicit bootstrap scripts.
- Model archives and extracted files are checked before use.
- ONNX is an executable graph format; use only manifest-verified models.
- OpenCV, ONNX Runtime, MMPose, and MMDetection use permissive project licenses, but their notices must ship with binaries where required.
- Model training datasets, including HALPE, HumanArt, and COCO-derived data, can impose separate terms. Weight redistribution is not approved.
- No Squatris project license has been selected.
