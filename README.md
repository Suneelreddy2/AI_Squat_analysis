# 🏋️ AI Barbell Squat Analysis & Skill Evaluator

A pure Python Computer Vision and Biomechanics assessment system grounded in Starting Strength barbell squat standards (*Document_for_skill.pdf*).

## 🌟 Key Features

- **📂 Video Analysis**: Dual-synced video player (Original vs AI Annotated) with one-click frame synchronization, rep-by-rep audit findings with PDF citations, and biomechanical plots.
- **📷 Live Camera Counter**: Real-time rep counting and biomechanical metrics (Knee Angle, Back Angle, Depth Status) via webcam using MediaPipe Pose.
- **📖 Document-Grounded Skill Engine**: 7 rule evaluations mapped directly to Starting Strength squat criteria with page citations.
- **📱 Responsive Top Navigation Bar**: Switch between Live mode and Video mode seamlessly, with YouTube link input or video file upload.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `ffmpeg` (installed on system)

### Installation

```bash
# Clone the repository
git clone https://github.com/CHANDU32455/HiringTests.git
cd HiringTests

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application
├── squat_skill/
│   ├── squat_rules.yaml        # Rule specifications with PDF citations
│   └── skill_engine.py         # Evaluator engine
├── vision/
│   ├── pose_detector.py        # MediaPipe Pose landmark extraction
│   ├── barbell_tracker.py      # Bar path tracking
│   ├── rep_detector.py         # Phase & rep segmentation
│   ├── biomechanics.py         # Angle & offset computations
│   ├── video_annotator.py      # Biomechanical overlay drawing
│   ├── processor.py            # Video processing pipeline
│   └── live_counter.py         # Real-time webcam rep counter
├── samples/
│   └── common_sample.mp4       # Pre-loaded sample squat video
└── requirements.txt            # Package dependencies
```
