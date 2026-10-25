"""
Comprehensive unit tests for VirtualLineCounter scenarios using simulated bounding-box trajectories.
"""
import sys
from fall_detection import VirtualLineCounter


def test_scenario_1_left_to_right():
    print("Test 1: One person crosses left to right...")
    counter = VirtualLineCounter(line_ratio=0.5, initial_count=0, allow_negative=False, deadband_pixels=20)
    width = 640

    # Frame 1: Person 1 starts on left
    counter.update(width, frame_index=1, detections=[(1, (100, 100, 140, 200))])
    assert counter.right_side_count == 0, f"Expected 0, got {counter.right_side_count}"
    assert counter.total_left_to_right == 0

    # Frame 2: Person 1 moves towards line (still on left side of deadband: 320 - 20 = 300)
    counter.update(width, frame_index=2, detections=[(1, (240, 100, 280, 200))])  # bottom-center x = 260
    assert counter.right_side_count == 0

    # Frame 3: Person 1 enters deadband (bottom-center x = 320)
    counter.update(width, frame_index=3, detections=[(1, (300, 100, 340, 200))])
    assert counter.right_side_count == 0

    # Frame 4: Person 1 emerges clearly onto right side (> 340)
    counter.update(width, frame_index=4, detections=[(1, (360, 100, 400, 200))])  # bottom-center x = 380
    assert counter.right_side_count == 1, f"Expected 1, got {counter.right_side_count}"
    assert counter.total_left_to_right == 1, f"Expected 1, got {counter.total_left_to_right}"
    print("  PASSED: Right-side count increased by 1.")


def test_scenario_2_right_to_left():
    print("Test 2: One person crosses right to left...")
    counter = VirtualLineCounter(line_ratio=0.5, initial_count=1, allow_negative=False, deadband_pixels=20)
    width = 640

    # Frame 1: Person 2 starts on right (bottom-center x = 450)
    counter.update(width, frame_index=1, detections=[(2, (430, 100, 470, 200))])
    assert counter.right_side_count == 1

    # Frame 2: Enters deadband
    counter.update(width, frame_index=2, detections=[(2, (310, 100, 330, 200))])  # center = 320
    assert counter.right_side_count == 1

    # Frame 3: Emerges on left side (< 300)
    counter.update(width, frame_index=3, detections=[(2, (180, 100, 220, 200))])  # center = 200
    assert counter.right_side_count == 0, f"Expected 0, got {counter.right_side_count}"
    assert counter.total_right_to_left == 1, f"Expected 1, got {counter.total_right_to_left}"
    print("  PASSED: Right-side count decreased by 1.")


def test_scenario_3_cross_and_cross_back():
    print("Test 3: One person crosses right, then crosses back left...")
    counter = VirtualLineCounter(line_ratio=0.5, initial_count=0, allow_negative=False, deadband_pixels=20)
    width = 640

    # Starts left
    counter.update(width, frame_index=1, detections=[(3, (100, 100, 140, 200))])
    # Crosses to right
    counter.update(width, frame_index=2, detections=[(3, (380, 100, 420, 200))])
    assert counter.right_side_count == 1
    assert counter.total_left_to_right == 1

    # Crosses back to left
    counter.update(width, frame_index=3, detections=[(3, (120, 100, 160, 200))])
    assert counter.right_side_count == 0, f"Expected 0, got {counter.right_side_count}"
    assert counter.total_left_to_right == 1
    assert counter.total_right_to_left == 1
    print("  PASSED: Net count change is zero after round trip.")


def test_scenario_4_hover_near_line():
    print("Test 4: Person stands near the line with jitter...")
    counter = VirtualLineCounter(line_ratio=0.5, initial_count=0, allow_negative=False, deadband_pixels=25)
    width = 640

    # Person starts at line edge
    counter.update(width, frame_index=1, detections=[(4, (280, 100, 320, 200))])  # center = 300 (left)
    # Oscillate right on the line boundary with jitter
    for f in range(2, 20):
        jitter_x = 310 + (f % 5) * 4  # between 310 and 326 (inside deadband [295, 345])
        counter.update(width, frame_index=f, detections=[(4, (jitter_x - 20, 100, jitter_x + 20, 200))])
        assert counter.right_side_count == 0, f"Spurious count on frame {f}"

    assert counter.total_left_to_right == 0
    assert counter.total_right_to_left == 0
    print("  PASSED: No repeated counting caused by bounding-box jitter near line.")


def test_scenario_5_multiple_people():
    print("Test 5: Multiple people cross in both directions...")
    counter = VirtualLineCounter(line_ratio=0.5, initial_count=2, allow_negative=False, deadband_pixels=20)
    width = 640

    # Frame 1: Person 10 on left, Person 20 on right
    counter.update(width, frame_index=1, detections=[
        (10, (100, 100, 140, 200)),  # left
        (20, (500, 100, 540, 200)),  # right
    ])
    assert counter.right_side_count == 2

    # Frame 2: Person 10 crosses to right, Person 20 crosses to left
    counter.update(width, frame_index=2, detections=[
        (10, (400, 100, 440, 200)),  # right (L->R)
        (20, (150, 100, 190, 200)),  # left  (R->L)
    ])
    # Net change: +1 -1 = 0
    assert counter.right_side_count == 2
    assert counter.total_left_to_right == 1
    assert counter.total_right_to_left == 1

    # Frame 3: Person 30 appears on left and crosses to right
    counter.update(width, frame_index=3, detections=[
        (30, (80, 100, 120, 200)),
    ])
    counter.update(width, frame_index=4, detections=[
        (30, (450, 100, 490, 200)),
    ])
    assert counter.right_side_count == 3
    assert counter.total_left_to_right == 2
    print("  PASSED: Multiple people crossing correctly tracked.")


def test_scenario_6_detection_loss():
    print("Test 6: Temporary detection loss does not duplicate count...")
    counter = VirtualLineCounter(line_ratio=0.5, initial_count=0, allow_negative=False, deadband_pixels=20)
    width = 640

    # Frame 1: Person on left
    counter.update(width, frame_index=1, detections=[(50, (100, 100, 140, 200))])
    assert counter.right_side_count == 0

    # Frame 2-5: Person temporarily lost (e.g. occlusion)
    for f in range(2, 6):
        counter.update(width, frame_index=f, detections=[])

    # Frame 6: Person reappears on right (same ID 50)
    counter.update(width, frame_index=6, detections=[(50, (420, 100, 460, 200))])
    assert counter.right_side_count == 1
    assert counter.total_left_to_right == 1

    # Frame 7: Continues on right
    counter.update(width, frame_index=7, detections=[(50, (450, 100, 490, 200))])
    assert counter.right_side_count == 1
    assert counter.total_left_to_right == 1
    print("  PASSED: Temporary detection loss did not cause duplicate crossing.")


def test_non_negative_boundary():
    print("Test: Non-negative count boundary...")
    # By default, right_side_count should clamp at 0
    counter = VirtualLineCounter(line_ratio=0.5, initial_count=0, allow_negative=False, deadband_pixels=20)
    width = 640

    counter.update(width, frame_index=1, detections=[(60, (500, 100, 540, 200))])  # right
    counter.update(width, frame_index=2, detections=[(60, (100, 100, 140, 200))])  # crosses left
    assert counter.right_side_count == 0, "Count should not be negative when allow_negative=False"
    assert counter.total_right_to_left == 1

    # When allow_negative=True
    counter_neg = VirtualLineCounter(line_ratio=0.5, initial_count=0, allow_negative=True, deadband_pixels=20)
    counter_neg.update(width, frame_index=1, detections=[(70, (500, 100, 540, 200))])
    counter_neg.update(width, frame_index=2, detections=[(70, (100, 100, 140, 200))])
    assert counter_neg.right_side_count == -1, "Count should allow negative when configured"
    assert counter_neg.total_right_to_left == 1
    print("  PASSED: Non-negative and negative count options operate as expected.")


if __name__ == "__main__":
    print("--- RUNNING VIRTUAL LINE COUNTER SCENARIO TESTS ---")
    test_scenario_1_left_to_right()
    test_scenario_2_right_to_left()
    test_scenario_3_cross_and_cross_back()
    test_scenario_4_hover_near_line()
    test_scenario_5_multiple_people()
    test_scenario_6_detection_loss()
    test_non_negative_boundary()
    print("\nALL 6 PEOPLE COUNTER SCENARIOS PASSED WITH 100% ACCURACY!")
