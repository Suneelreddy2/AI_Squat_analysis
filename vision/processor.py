import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Callable

from vision.pose_detector import PoseDetector
from vision.barbell_tracker import BarbellTracker
from vision.rep_detector import RepetitionDetector
from vision.biomechanics import BiomechanicsCalculator
from vision.video_annotator import VideoAnnotator
from squat_skill.skill_engine import SquatSkillEngine

class SquatVideoProcessor:
    def __init__(self, rules_path: str = None):
        self.skill_engine = SquatSkillEngine(rules_path=rules_path)

    def process_video(
        self,
        video_path: str,
        output_dir: str = "output",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes a squat video end-to-end, performs landmark tracking, rep segmentation,
        rule evaluation against squat_skill, and renders annotated playable MP4 video.
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pose_detector = PoseDetector()
        bar_tracker = BarbellTracker()

        frames_telemetry = []
        raw_frames = []
        landmarks_history = []
        bar_history = []

        if progress_callback:
            progress_callback(0.05, "Extracting landmarks & tracking barbell...")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, conf = pose_detector.process_frame(frame)
            bar_pos, bar_conf = bar_tracker.track(frame, landmarks)

            frame_telemetry = BiomechanicsCalculator.compute_frame_telemetry(landmarks, bar_pos)
            frame_telemetry["frame_idx"] = frame_idx
            frame_telemetry["timestamp"] = frame_idx / fps
            frame_telemetry["pose_confidence"] = conf
            frame_telemetry["bar_tracked"] = bar_conf > 0.4

            frames_telemetry.append(frame_telemetry)
            raw_frames.append(frame)
            landmarks_history.append(landmarks)
            bar_history.append(bar_pos)

            frame_idx += 1
            if progress_callback and frame_idx % 15 == 0 and total_frames > 0:
                prog = 0.05 + 0.45 * (frame_idx / total_frames)
                progress_callback(prog, f"Processing frame {frame_idx}/{total_frames}...")

        cap.release()
        pose_detector.close()

        if progress_callback:
            progress_callback(0.55, "Detecting repetition phases & bottom positions...")

        # Segment repetitions
        rep_detector = RepetitionDetector(fps=fps)
        repetitions = rep_detector.detect_repetitions(frames_telemetry)

        # Evaluate each repetition against SquatSkill rules
        evaluated_reps = []
        for rep in repetitions:
            start_f = rep["start_frame"]
            bottom_f = rep["bottom_frame"]
            end_f = rep["end_frame"]

            bottom_telemetry = frames_telemetry[bottom_f] if bottom_f < len(frames_telemetry) else {}
            
            # Compute rep aggregate telemetry
            rep_frames = frames_telemetry[start_f:end_f+1]
            valid_rep_frames = [f for f in rep_frames if f.get("valid", False)]
            
            max_bar_dev = max([f.get("bar_dev_px", 0) for f in valid_rep_frames], default=0, key=abs)
            max_norm_bar_dev = max([f.get("norm_bar_dev", 0) for f in valid_rep_frames], default=0)
            avg_conf = np.mean([f.get("pose_confidence", 1.0) for f in valid_rep_frames]) if valid_rep_frames else 0.0
            
            # Ascent back angle stability check
            ascent_frames = frames_telemetry[bottom_f:end_f+1]
            ascent_back_angles = [f.get("back_angle", 45) for f in ascent_frames if f.get("valid", False)]
            min_ascent_back = min(ascent_back_angles, default=45.0)
            bottom_back = bottom_telemetry.get("back_angle", 45.0)
            ascent_change = abs(bottom_back - min_ascent_back)

            rep_summary_telemetry = {
                "pose_confidence": avg_conf,
                "depth_delta_px": bottom_telemetry.get("depth_delta_px", 0.0),
                "is_deep": bottom_telemetry.get("is_deep", False),
                "max_bar_dev_px": max_bar_dev,
                "normalized_bar_dev": max_norm_bar_dev,
                "bar_tracked": bottom_telemetry.get("bar_tracked", True),
                "bottom_back_angle": bottom_back,
                "ascent_back_angle_change": ascent_change,
                "knee_forward_over_toe_px": bottom_telemetry.get("knee_forward_px", 0.0),
                "gaze_angle_deg": bottom_telemetry.get("gaze_angle", -15.0),
                "hip_drive_initiated": True
            }

            findings = self.skill_engine.evaluate_repetition(rep_summary_telemetry)

            rep_eval = {
                **rep,
                "bottom_timestamp": bottom_f / fps,
                "telemetry": rep_summary_telemetry,
                "findings": findings
            }
            evaluated_reps.append(rep_eval)

        if progress_callback:
            progress_callback(0.70, "Rendering annotated video with biomechanical overlays...")

        # Annotate video frames and write to MP4 file
        annotator = VideoAnnotator()
        annotated_filename = f"annotated_{os.path.basename(video_path)}"
        if not annotated_filename.endswith(".mp4"):
            annotated_filename = os.path.splitext(annotated_filename)[0] + ".mp4"
        annotated_path = os.path.join(output_dir, annotated_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        for idx, frame in enumerate(raw_frames):
            lm = landmarks_history[idx]
            tel = frames_telemetry[idx]
            
            # Determine current rep & phase
            cur_rep = None
            cur_phase = "SETUP / STANDING"

            for rep in evaluated_reps:
                if rep["start_frame"] <= idx <= rep["end_frame"]:
                    cur_rep = rep
                    if abs(idx - rep["bottom_frame"]) <= 2:
                        cur_phase = "BOTTOM"
                    elif idx < rep["bottom_frame"]:
                        cur_phase = "DESCENT"
                    else:
                        cur_phase = "ASCENT"
                    break

            annotated_frame = annotator.annotate_frame(
                frame=frame,
                landmarks=lm,
                telemetry=tel,
                bar_history=bar_history[:idx+1],
                current_rep=cur_rep,
                rep_phase=cur_phase
            )
            out_writer.write(annotated_frame)

        out_writer.release()

        # Convert video to H.264 mp4 using ffmpeg if available for browser compatibility
        web_compatible_path = os.path.join(output_dir, f"web_{annotated_filename}")
        ffmpeg_cmd = f"ffmpeg -y -i '{annotated_path}' -vcodec libx264 -pix_fmt yuv420p '{web_compatible_path}' >/dev/null 2>&1"
        res = os.system(ffmpeg_cmd)
        final_video_path = web_compatible_path if res == 0 and os.path.exists(web_compatible_path) else annotated_path

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        return {
            "video_path": video_path,
            "annotated_video_path": final_video_path,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": total_frames / fps,
            "repetitions": evaluated_reps,
            "frames_telemetry": frames_telemetry
        }
