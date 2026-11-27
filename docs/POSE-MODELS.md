# Pose models

## Initial selection

Phase 0A uses YOLOX-tiny HumanArt at 416×416 for person detection and RTMPose-m at 256×192 trained with the HALPE-26 layout for pose estimation.

RTMPose-m is a middle-sized starting point: upstream reports materially higher accuracy than the tiny/small variants while retaining a real-time design. HALPE-26 was selected instead of COCO-17 because it provides heels and foot landmarks needed to evaluate squat and stance quality later. YOLOX-tiny reduces the detector portion of latency while retaining an explicit top-down region-of-interest step.

This is an experimental baseline, not a permanent model decision. Phase 0B will compare it with RTMW, MediaPipe Heavy, and any viable Maxine path under the same recorded inputs.

## Reproducible artifacts

`models/manifest.json` is the single authoritative artifact definition. It records source URLs, filenames, upstream versions, input/output contracts, SHA-256 checksums, licensing notes, and redistribution status. `scripts/acquire-models.ps1` consumes that manifest, reuses already verified files, verifies archives and extracted ONNX files, and retains only the named model outputs.

Weights are excluded from Git. Redistribution has not been approved.

## Pipeline contract

The detector performs aspect-preserving resize into a top-left-aligned 416×416 BGR canvas padded with 114. Its exported NMS produces `dets` and `labels`; the highest-confidence person above the configured threshold is selected.

The person box is padded by 1.25 and adjusted to the 192:256 pose aspect ratio. An affine transform creates the pose crop. The crop is converted BGR-to-RGB and normalized with mean `(123.675, 116.28, 103.53)` and standard deviation `(58.395, 57.12, 57.375)`.

RTMPose returns x and y SimCC distributions for 26 keypoints. Each axis maximum is decoded with split ratio 2.0, mapped through the crop center/scale into source pixels, normalized, and mapped from HALPE-26 to canonical anatomy. Invalid and below-threshold landmarks become unavailable.

Session initialization requires the pinned input and output names, tensor ranks, and element types. The export declares RTMPose output axes dynamically, so each inference additionally requires runtime shapes `[1,26,384]` and `[1,26,512]` and exact element counts before tensor memory is read. Detector outputs must be float32 `[1,N,5]` detections and int64 `[1,N]` labels with matching `N`.

The implementation follows `deploy.json`, `detail.json`, and `pipeline.json` embedded in the pinned official OpenMMLab archives and was cross-checked against rtmlib revision `5ea82df885d6f2a66923f1de2d15c799af0ed732` for center/scale, affine, RGB normalization, SimCC decoding, mean confidence, and inverse coordinate transformation. Deterministic fixtures cover region aspect, RGB channel order and normalization, argmax/mean confidence, inverse coordinates, malformed shapes, and non-finite values.

## Limitations

- The detector selects one player by score and does not maintain identity.
- Model confidence is not calibrated.
- Estimated depth is absent for this 2D model.
- Physical accuracy, occlusion behavior, and camera-distance limits remain unqualified.
- Training dataset terms require legal review before distributing model weights.
