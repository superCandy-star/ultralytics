---
name: hand-anchored-tracking-design
description: "Hand-anchored object tracking for smart fridge: YOLOv11 + ByteTrack with hand as spatial anchor to solve obj detection failures"
metadata: 
  node_type: memory
  type: project
  originSessionId: b2f24333-96ee-41eb-b69b-058de4cdb4ef
---

# Hand-Anchored Obj Tracking for Smart Fridge

Discussed on 2026-06-29.

## Background
Smart fridge peripheral device. YOLOv11m detects 2 classes: hand and obj (hand-held object). Obj is a virtual superclass covering 8-9 food categories (apple, tomato, yogurt, bacon, etc.). ByteTrack handles tracking. Hand tracking works well; obj suffers frequent missed detections due to occlusion and feature diversity.

## Root Cause of Obj Missed Detections
- Not data quantity (5K+ images), but **occlusion fragmentation**: hand grasping fragments obj into disconnected regions, detection features break, NMS eats weak detections
- **Feature coupling**: hand and obj boxes overlap heavily, training gradient competition, obj often loses
- Obj intra-class variance is large (shape, material, color, size span is huge) despite only being 8-9 subclasses

## Solution: Hand-Anchored Tracking (designed but not yet implemented)

### Core idea
Hand serves as a third-party spatial anchor. Obj track association no longer relies on IoU between obj prediction and obj detection; instead uses hand position as a bridge.

### Strategy
1. Each obj track binds to a hand_track_id at creation (nearest hand in space)
2. When obj is missed: propagate obj position using hand's motion vector (obj moves with hand as quasi-rigid body)
3. When obj reappears: match by checking "is there an obj detection near the bound hand's predicted position?" — no IoU matching needed
4. On re-association: KF measurement update, R can be temporarily scaled (e.g. ×2) as buffer
5. OC-SORT ORU can be used on re-activation to compress inflated covariance

### Maximum tolerance
Set a max missed-frames threshold (e.g. 15 frames @ 30fps). Beyond that, terminate track, new detection starts fresh track.

### Key edge cases
- One hand may have multiple obj detections nearby → pick highest confidence
- Hand-obj binding is persistent (single-hand operation assumed for fridge scenario)
- "Near hand" uses center-distance threshold, not IoU (more lenient)

## Related Memory
- [[ocsort-paper-discussion]] — OC-SORT ORU theory applicable to covariance recovery on re-activation
- [[chat-formula-preference]] — Use plain text for math in chat
