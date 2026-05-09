"""
Kinexis — Per-Class Dry-Run Tests
Run this before inference.py to confirm everything loads and works.

Usage:
    python tests/test_dryrun.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import collections
import math
import numpy as np
import torch
import torch.nn.functional as F

from inference import (
    STGCNFineTuned, SlidingWindowBuffer, RepCounter, SessionAccuracyTracker,
    get_angles, get_form_feedback, classify, normalise_skeleton, load_model,
    A3_np, CLIP_LEN, NUM_JOINTS, NUM_CLASSES, LABEL_MAP, REP_RULES,
    STATE_UP, STATE_DOWN, STATE_INIT,
    HEAD, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI,
    L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK,
)

PASS, FAIL = '✓', '✗'

# ── Synthetic skeleton builder ─────────────────────────────────────────────────
NEUTRAL = {
    HEAD:  [200, 100],
    L_SHO: [180, 150], R_SHO: [220, 150],
    L_ELB: [160, 200], R_ELB: [240, 200],
    L_WRI: [150, 250], R_WRI: [250, 250],
    L_HIP: [185, 270], R_HIP: [215, 270],
    L_KNE: [180, 330], R_KNE: [220, 330],
    L_ANK: [175, 390], R_ANK: [225, 390],
}

def make_skeleton(overrides=None):
    px = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    for idx, coord in NEUTRAL.items():
        px[idx] = coord
    for idx, coord in (overrides or {}).items():
        px[idx] = coord
    return px

# Down/up pose overrides per exercise
POSES = {
    'squats': {
        'down': {L_KNE:[200,300], L_HIP:[200,200], L_ANK:[200,380],
                 R_KNE:[250,300], R_HIP:[250,200], R_ANK:[250,380]},
        'up':   {L_KNE:[200,265], L_HIP:[200,200], L_ANK:[200,380],
                 R_KNE:[250,265], R_HIP:[250,200], R_ANK:[250,380]},
    },
    'push_ups': {
        'down': {L_ELB:[150,220], L_SHO:[130,200], L_WRI:[170,240],
                 R_ELB:[250,220], R_SHO:[270,200], R_WRI:[230,240]},
        'up':   {L_ELB:[150,230], L_SHO:[130,200], L_WRI:[170,195],
                 R_ELB:[250,230], R_SHO:[270,200], R_WRI:[230,195]},
    },
    'bench_press': {
        'down': {L_ELB:[150,220], L_SHO:[130,200], L_WRI:[170,240],
                 R_ELB:[250,220], R_SHO:[270,200], R_WRI:[230,240]},
        'up':   {L_ELB:[150,230], L_SHO:[130,200], L_WRI:[170,195],
                 R_ELB:[250,230], R_SHO:[270,200], R_WRI:[230,195]},
    },
    'pull_ups': {
        'down': {L_ELB:[150,230], L_SHO:[130,200], L_WRI:[170,195],
                 R_ELB:[250,230], R_SHO:[270,200], R_WRI:[230,195]},
        'up':   {L_ELB:[150,220], L_SHO:[130,200], L_WRI:[170,240],
                 R_ELB:[250,220], R_SHO:[270,200], R_WRI:[230,240]},
    },
    'sit_ups': {
        'down': {L_HIP:[200,250], L_SHO:[200,360], L_KNE:[200,200],
                 R_HIP:[250,250], R_SHO:[250,360], R_KNE:[250,200]},
        'up':   {L_HIP:[200,250], L_SHO:[200,285], L_KNE:[200,200],
                 R_HIP:[250,250], R_SHO:[250,285], R_KNE:[250,200]},
    },
    'jumping_jacks': {
        'down': {L_SHO:[180,220], L_ELB:[165,245], L_HIP:[200,300],
                 R_SHO:[220,220], R_ELB:[235,245], R_HIP:[200,300]},
        'up':   {L_SHO:[180,220], L_ELB:[138,198], L_HIP:[200,300],
                 R_SHO:[220,220], R_ELB:[262,198], R_HIP:[200,300]},
    },
    'jump_rope': {
        'down': {L_KNE:[200,270], L_HIP:[200,200], L_ANK:[200,380],
                 R_KNE:[250,270], R_HIP:[250,200], R_ANK:[250,380]},
        'up':   {L_KNE:[200,258], L_HIP:[200,200], L_ANK:[200,380],
                 R_KNE:[250,258], R_HIP:[250,200], R_ANK:[250,380]},
    },
}

# ── Tests ─────────────────────────────────────────────────────────────────────
def test_buffer_dtype_shape():
    """Buffer must produce float32 tensors of correct shape."""
    buf  = SlidingWindowBuffer(CLIP_LEN)
    px   = make_skeleton()
    conf = np.ones(NUM_JOINTS, dtype=np.float32)
    for _ in range(CLIP_LEN):
        buf.push(normalise_skeleton(px), conf)
    xy_t, conf_t = buf.get_tensors()
    assert xy_t.dtype   == torch.float32,                  f'xy dtype: {xy_t.dtype}'
    assert conf_t.dtype == torch.float32,                  f'conf dtype: {conf_t.dtype}'
    assert tuple(xy_t.shape)   == (1,2,CLIP_LEN,NUM_JOINTS), f'xy shape: {xy_t.shape}'
    assert tuple(conf_t.shape) == (1,1,CLIP_LEN,NUM_JOINTS), f'conf shape: {conf_t.shape}'
    return True

def test_model_loads():
    """Model loads from checkpoint without errors."""
    model = load_model()
    assert next(model.parameters()).dtype == torch.float32
    return True, model

def test_classify(model):
    """classify() returns a valid class index and float confidence."""
    buf  = SlidingWindowBuffer(CLIP_LEN)
    conf = np.ones(NUM_JOINTS, dtype=np.float32)
    px   = make_skeleton()
    for _ in range(CLIP_LEN):
        buf.push(normalise_skeleton(px), conf)
    cls_id, confidence = classify(model, buf)
    assert cls_id is not None,                  'classify() returned None'
    assert cls_id in LABEL_MAP,                 f'cls_id {cls_id} not in LABEL_MAP'
    assert 0.0 <= confidence <= 1.0,            f'confidence out of range: {confidence}'
    return True

def test_angles_in_range(exercise):
    """get_angles() returns values in [0, 180] for all exercises."""
    px     = make_skeleton(POSES.get(exercise, {}).get('down'))
    angles = get_angles(px)
    for k, v in angles.items():
        assert 0 <= v <= 180, f'{exercise} angle {k}={v:.1f} out of range'
    return True, angles

def test_rep_counter(exercise):
    """Rep counter transitions correctly between down/up poses."""
    poses      = POSES.get(exercise, {})
    down_px    = make_skeleton(poses.get('down'))
    up_px      = make_skeleton(poses.get('up'))
    rc         = RepCounter()
    angles_d   = get_angles(down_px)
    angles_u   = get_angles(up_px)
    rc.update(exercise, angles_d)
    state_after_down = rc.states[exercise]
    rc.update(exercise, angles_u)
    state_after_up   = rc.states[exercise]
    reps             = rc.reps[exercise]
    # State must have changed from INIT at minimum
    assert state_after_down != STATE_INIT or state_after_up != STATE_INIT, \
        f'{exercise}: state machine never left INIT'
    return True, state_after_down, state_after_up, reps

def test_feedback(exercise):
    """get_form_feedback() returns a non-empty list of (severity, str) tuples."""
    poses   = POSES.get(exercise, {})
    down_px = make_skeleton(poses.get('down'))
    angles  = get_angles(down_px)
    fb      = get_form_feedback(exercise, angles, STATE_DOWN)
    assert isinstance(fb, list) and len(fb) > 0,    f'{exercise}: feedback empty'
    assert all(isinstance(sev, str) and isinstance(msg, str)
               for sev, msg in fb),                 f'{exercise}: feedback malformed'
    return True, fb

def test_accuracy_tracker():
    """SessionAccuracyTracker computes accuracy correctly."""
    at = SessionAccuracyTracker(ground_truth='squats')
    at.update('squats', 0.9)
    at.update('squats', 0.8)
    at.update('push_ups', 0.7)
    acc, _ = at.accuracy()
    assert abs(acc - 2/3) < 1e-6, f'Expected 0.667, got {acc}'
    return True

# ── Runner ────────────────────────────────────────────────────────────────────
def run_all():
    print('=' * 55)
    print('  Kinexis — Per-Class Dry-Run Tests')
    print('=' * 55)

    results  = []
    all_pass = True

    # ── Global tests ──────────────────────────────────────────────────────────
    print('\n── Global ──────────────────────────────────────────')

    try:
        test_buffer_dtype_shape()
        print(f'  {PASS} Buffer dtype & shape')
    except AssertionError as e:
        print(f'  {FAIL} Buffer dtype & shape — {e}'); all_pass = False

    try:
        ok, model = test_model_loads()
        print(f'  {PASS} Model loads from checkpoint')
    except Exception as e:
        print(f'  {FAIL} Model loads — {e}'); all_pass = False; model = None

    if model:
        try:
            test_classify(model)
            print(f'  {PASS} classify() runs without error')
        except AssertionError as e:
            print(f'  {FAIL} classify() — {e}'); all_pass = False

    try:
        test_accuracy_tracker()
        print(f'  {PASS} SessionAccuracyTracker')
    except AssertionError as e:
        print(f'  {FAIL} SessionAccuracyTracker — {e}'); all_pass = False

    # ── Per-class tests ───────────────────────────────────────────────────────
    print('\n── Per-Class ───────────────────────────────────────')

    for exercise in LABEL_MAP.values():
        print(f'\n  {exercise.upper()}')
        ok_all = True

        # Angles
        try:
            _, angles = test_angles_in_range(exercise)
            rule      = REP_RULES.get(exercise, {})
            key_a     = {k: f'{angles[k]:.0f}°' for k in rule.get('keys',[]) if k in angles}
            print(f'    {PASS} angles  {key_a}')
        except AssertionError as e:
            print(f'    {FAIL} angles — {e}'); ok_all = False

        # Rep counter
        try:
            _, s_down, s_up, reps = test_rep_counter(exercise)
            print(f'    {PASS} rep counter  down={s_down} → up={s_up}  reps={reps}')
        except AssertionError as e:
            print(f'    {FAIL} rep counter — {e}'); ok_all = False

        # Form feedback
        try:
            _, fb = test_feedback(exercise)
            print(f'    {PASS} feedback  {fb}')
        except (AssertionError, Exception) as e:
            print(f'    {FAIL} feedback — {e}'); ok_all = False

        # Classify (per-class)
        if model:
            try:
                buf  = SlidingWindowBuffer(CLIP_LEN)
                conf = np.ones(NUM_JOINTS, dtype=np.float32)
                down_px = make_skeleton(POSES.get(exercise, {}).get('down'))
                up_px   = make_skeleton(POSES.get(exercise, {}).get('up', POSES.get(exercise, {}).get('down')))
                for i in range(CLIP_LEN):
                    px = down_px if i % 2 == 0 else up_px
                    buf.push(normalise_skeleton(px), conf)
                cls_id, confidence = classify(model, buf)
                predicted = LABEL_MAP[cls_id] if cls_id is not None else 'none'
                match = '✓ match' if predicted == exercise else f'≠ got {predicted}'
                print(f'    {PASS} classify  conf={confidence*100:.1f}%  {match}')
            except Exception as e:
                print(f'    {FAIL} classify — {e}'); ok_all = False

        results.append((exercise, ok_all))
        if not ok_all: all_pass = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*55}')
    print(f'  {"Exercise":<18} {"Result"}')
    print(f'{"─"*55}')
    for exercise, ok in results:
        print(f'  {exercise:<18} {PASS if ok else FAIL}')
    print(f'{"="*55}')
    print(f'\n  {"All tests passed! ✓" if all_pass else "Some tests failed — check above."}')
    print(f'\n  Note: classify ≠ match is expected for synthetic skeletons.')
    print(f'  Run inference.py on real video for true accuracy.\n')

    return all_pass


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
