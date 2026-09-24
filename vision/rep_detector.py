import numpy as np
from typing import List, Dict, Any

class RepetitionDetector:
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def detect_repetitions(self, frames_telemetry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes frame-by-frame telemetry to detect squat repetitions.
        Returns a list of repetition dictionary metadata.
        """
        if not frames_telemetry:
            return []

        hip_y_series = []
        valid_indices = []
        for i, data in enumerate(frames_telemetry):
            if data.get("valid", False) and "hip_y" in data:
                hip_y_series.append(data["hip_y"])
                valid_indices.append(i)

        if len(hip_y_series) < 10:
            return []

        smoothed_y = np.convolve(hip_y_series, np.ones(7)/7, mode='same')

        y_min = float(np.min(smoothed_y))
        y_max = float(np.max(smoothed_y))
        y_range = y_max - y_min

        # Threshold for meaningful squat movement: at least 20px vertical change
        if y_range < 20.0:
            return []

        # Find local peaks of hip Y (bottom turnaround point of squat)
        bottom_candidates = []
        for i in range(3, len(smoothed_y) - 3):
            val = smoothed_y[i]
            if val >= smoothed_y[i-1] and val >= smoothed_y[i-2] and val >= smoothed_y[i+1] and val >= smoothed_y[i+2]:
                if (val - y_min) > 0.30 * y_range:
                    bottom_candidates.append(i)

        if not bottom_candidates:
            max_idx = int(np.argmax(smoothed_y))
            if (smoothed_y[max_idx] - y_min) > 0.30 * y_range:
                bottom_candidates = [max_idx]

        # Merge close bottom candidates
        min_dist = int(self.fps * 0.8)
        merged_bottoms = []
        for b in bottom_candidates:
            if not merged_bottoms or (b - merged_bottoms[-1]) > min_dist:
                merged_bottoms.append(b)
            else:
                if smoothed_y[b] > smoothed_y[merged_bottoms[-1]]:
                    merged_bottoms[-1] = b

        repetitions = []
        for idx, bottom_idx in enumerate(merged_bottoms):
            start_rel_idx = 0
            for k in range(bottom_idx - 1, -1, -1):
                if (smoothed_y[k] - y_min) <= 0.15 * y_range:
                    start_rel_idx = k
                    break
                if k > 0 and smoothed_y[k] <= smoothed_y[k-1] and smoothed_y[k] <= smoothed_y[k+1]:
                    start_rel_idx = k
                    break

            end_rel_idx = len(smoothed_y) - 1
            for k in range(bottom_idx + 1, len(smoothed_y)):
                if (smoothed_y[k] - y_min) <= 0.15 * y_range:
                    end_rel_idx = k
                    break
                if k < len(smoothed_y) - 1 and smoothed_y[k] <= smoothed_y[k-1] and smoothed_y[k] <= smoothed_y[k+1]:
                    end_rel_idx = k
                    break

            start_frame = valid_indices[start_rel_idx]
            bottom_frame = valid_indices[bottom_idx]
            end_frame = valid_indices[end_rel_idx]

            duration = (end_frame - start_frame) / max(self.fps, 1.0)
            
            repetitions.append({
                "rep_number": len(repetitions) + 1,
                "start_frame": start_frame,
                "bottom_frame": bottom_frame,
                "end_frame": end_frame,
                "duration_sec": duration,
                "descent_duration_sec": (bottom_frame - start_frame) / max(self.fps, 1.0),
                "ascent_duration_sec": (end_frame - bottom_frame) / max(self.fps, 1.0)
            })

        return repetitions
