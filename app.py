import os
import sys
import base64
import tempfile
import time
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.processor import SquatVideoProcessor
from squat_skill.skill_engine import SquatSkillEngine
from vision.live_counter import LiveRepCounter

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Barbell Squat Analysis",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Dark Styles ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], .main { background: #080d1a !important; }
* { font-family: 'Inter', sans-serif; }
h1,h2,h3,h4,h5 { font-family: 'Outfit', sans-serif !important; }

[data-testid="stMetric"] {
    background: rgba(20,30,50,0.7);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
}
.card {
    background: rgba(15, 22, 40, 0.75);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
}
.status-badge-pass   { background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.35); border-radius:20px; padding:4px 14px; font-weight:600; font-size:.82rem; }
.status-badge-fail   { background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid rgba(239,68,68,0.35);  border-radius:20px; padding:4px 14px; font-weight:600; font-size:.82rem; }
.status-badge-cannot { background:rgba(148,163,184,0.12);color:#94a3b8; border:1px solid rgba(148,163,184,0.3); border-radius:20px; padding:4px 14px; font-weight:600; font-size:.82rem; }
.citation-pill { background:rgba(99,102,241,0.14); color:#818cf8; border:1px solid rgba(99,102,241,0.3); border-radius:8px; padding:3px 9px; font-size:.74rem; font-weight:500; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def video_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def synced_video_player(original_path: str, annotated_path: str) -> None:
    """Full-width synchronized dual video player — one play button controls both."""
    orig_b64 = video_b64(original_path)
    ann_b64  = video_b64(annotated_path)

    html = f"""
<!DOCTYPE html><html><head>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#080d1a; font-family:'Inter',sans-serif; overflow:hidden; }}
.stage {{ display:flex; flex-direction:column; width:100%; gap:10px; }}
.labels {{ display:flex; width:100%; gap:8px; }}
.label {{ flex:1; text-align:center; color:#94a3b8; font-size:.78rem; font-weight:600;
          letter-spacing:.06em; text-transform:uppercase; padding:4px 0 2px; }}
.videos {{ display:flex; width:100%; gap:8px; align-items:flex-start; }}
.vid-wrap {{ flex:1; background:#000; border-radius:10px; overflow:hidden;
             border:1px solid rgba(255,255,255,0.08); }}
video {{ width:100%; height:auto; display:block; object-fit:contain; max-height:70vh; }}
.controls {{ display:flex; align-items:center; gap:14px;
             background:rgba(15,22,40,0.85); border:1px solid rgba(255,255,255,0.07);
             border-radius:12px; padding:10px 18px; backdrop-filter:blur(12px); }}
.btn {{ background:linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; border:none;
        border-radius:8px; padding:8px 20px; font-size:.88rem; font-weight:600;
        cursor:pointer; transition:opacity .15s; white-space:nowrap; }}
.btn:hover {{ opacity:.85; }}
.seek {{ flex:1; -webkit-appearance:none; height:5px; border-radius:3px;
         background:#1e2a40; outline:none; cursor:pointer; accent-color:#6366f1; }}
.time-lbl {{ color:#64748b; font-size:.78rem; white-space:nowrap; min-width:90px; text-align:right; }}
.mute-btn {{ background:rgba(99,102,241,0.15); color:#818cf8;
             border:1px solid rgba(99,102,241,0.3); border-radius:8px;
             padding:7px 14px; font-size:.8rem; cursor:pointer; }}
.mute-btn:hover {{ opacity:.8; }}
</style>
</head><body>
<div class="stage">
  <div class="labels">
    <div class="label">📹 Original Input</div>
    <div class="label">🤖 AI Annotated Analysis</div>
  </div>
  <div class="videos">
    <div class="vid-wrap">
      <video id="v1" preload="auto" playsinline muted>
        <source src="data:video/mp4;base64,{orig_b64}" type="video/mp4">
      </video>
    </div>
    <div class="vid-wrap">
      <video id="v2" preload="auto" playsinline muted>
        <source src="data:video/mp4;base64,{ann_b64}" type="video/mp4">
      </video>
    </div>
  </div>
  <div class="controls">
    <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play Both</button>
    <input  class="seek" type="range" id="seeker" min="0" max="1000" value="0"
            oninput="onSeek(this.value)">
    <span   class="time-lbl" id="timeLbl">0.0 / 0.0 s</span>
    <button class="mute-btn" id="muteBtn" onclick="toggleMute()">🔇 Mute</button>
  </div>
</div>
<script>
const v1=document.getElementById('v1'),v2=document.getElementById('v2');
const btn=document.getElementById('playBtn'),seek=document.getElementById('seeker');
const tLbl=document.getElementById('timeLbl'),mBtn=document.getElementById('muteBtn');
let muted=true;

function syncTick(){{
  if(!v1.paused){{
    const drift=v1.currentTime-v2.currentTime;
    if(Math.abs(drift)>0.05) v2.currentTime=v1.currentTime;
    seek.value=Math.round((v1.currentTime/(v1.duration||1))*1000);
    tLbl.textContent=v1.currentTime.toFixed(1)+' / '+(v1.duration||0).toFixed(1)+' s';
  }}
  requestAnimationFrame(syncTick);
}}
requestAnimationFrame(syncTick);

function togglePlay(){{
  if(v1.paused){{v1.play();v2.play();btn.textContent='⏸ Pause';}}
  else{{v1.pause();v2.pause();btn.textContent='▶ Play Both';}}
}}
function onSeek(val){{
  const t=(val/1000)*(v1.duration||0);
  v1.currentTime=t; v2.currentTime=t;
  tLbl.textContent=t.toFixed(1)+' / '+(v1.duration||0).toFixed(1)+' s';
}}
function toggleMute(){{
  muted=!muted; v1.muted=muted; v2.muted=muted;
  mBtn.textContent=muted?'🔇 Mute':'🔊 Sound';
}}
[v1,v2].forEach(v=>v.addEventListener('ended',()=>{{v1.pause();v2.pause();btn.textContent='▶ Play Both';}}));
document.addEventListener('keydown',e=>{{if(e.code==='Space'){{e.preventDefault();togglePlay();}}}});
</script>
</body></html>
"""
    components.html(html, height=720, scrolling=False)


# ── WebRTC Transformer (defined before any st.* calls) ────────────────────────
class _SquatTransformer(VideoProcessorBase):
    """Processes webcam frames server-side: MediaPipe pose + rep counting."""
    def __init__(self):
        self.counter = LiveRepCounter()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        out = self.counter.process_frame(bgr)
        return av.VideoFrame.from_ndarray(out, format="bgr24")


def _render_live_camera():
    """Live webcam panel — WebRTC stream + real-time KPI cards."""
    st.markdown("""
    <h3 style='margin-bottom:4px;background:linear-gradient(135deg,#6366f1,#38bdf8);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
      📷 Live Rep Counter — Laptop Camera
    </h3>
    <p style='color:#64748b;font-size:.88rem;margin-top:0;'>
      Stand <strong>sideways</strong> to your camera (left or right side facing it).
      MediaPipe tracks your body frame-by-frame and counts reps in real time.
    </p>
    """, unsafe_allow_html=True)

    st.info(
        "**Tip:** Ensure your full body (head → feet) is visible sideways in frame. "
        "Reps increment when your hips rise back to standing height after each squat descent."
    )

    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="squat-live",
        video_processor_factory=_SquatTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.divider()
    st.caption("📊 Live Biomechanics — updates every 0.5 s while camera is active")

    k1, k2, k3, k4, k5 = st.columns(5)
    slot_reps  = k1.empty()
    slot_phase = k2.empty()
    slot_knee  = k3.empty()
    slot_back  = k4.empty()
    slot_depth = k5.empty()

    # Defaults while not playing
    slot_reps.metric("🔁 Reps",      "0")
    slot_phase.metric("🧩 Phase",    "⏸ STANDING")
    slot_knee.metric("🦵 Knee",      "—")
    slot_back.metric("📐 Back",      "—")
    slot_depth.metric("📉 Depth OK", "—")

    if st.button("🔄 Reset Counter", key="live_reset_btn"):
        if ctx.video_processor:
            ctx.video_processor.counter.reset()

    if ctx.state.playing:
        phase_map = {
            "BOTTOM":   "🟢 BOTTOM",
            "DESCENT":  "🔽 DESCENT",
            "ASCENT":   "🔼 ASCENT",
            "STANDING": "⏸ STANDING",
        }
        while ctx.state.playing:
            time.sleep(0.5)
            if not ctx.video_processor:
                break
            m = ctx.video_processor.counter.get_metrics()
            slot_reps.metric("🔁 Reps",      str(m['rep_count']))
            slot_phase.metric("🧩 Phase",     phase_map.get(m['phase'], m['phase']))
            slot_knee.metric("🦵 Knee",       f"{m['knee_angle']:.0f}°")
            slot_back.metric("📐 Back",       f"{m['back_angle']:.0f}°")
            slot_depth.metric("📉 Depth OK",  "✅ Yes" if m['depth_ok'] else "❌ Not yet")


# ── Top Navigation & Control Bar ─────────────────────────────────────────────
st.markdown("""
<style>
.top-navbar {
    background: rgba(15, 22, 40, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}
.brand-title {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}
.brand-sub {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# Top Banner Row: Brand Title + Mode Segment Switch
header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    st.markdown("""
    <div>
      <div class="brand-title">🏋️ AI Barbell Squat Analysis & Skill Evaluator</div>
      <div class="brand-sub">Pure Python Computer Vision · MediaPipe Pose · Grounded in Starting Strength Standards</div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    app_mode = st.radio(
        "Mode Switch",
        options=["📂 Video Analysis", "📷 Live Camera Counter"],
        horizontal=True,
        label_visibility="collapsed",
        key="app_mode_switch"
    )

selected_video_path = None
run_analysis = False

# Interactive Control Toolbar for Video Analysis mode
if app_mode == "📂 Video Analysis":
    st.markdown("""
    <div style='background:rgba(20,30,52,0.6);border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:14px 20px;margin-bottom:20px;backdrop-filter:blur(10px);'>
    """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 2.5, 1.2])

    with ctrl_col1:
        video_source_type = st.selectbox(
            "Select Video Source:",
            options=["🎥 YouTube Video (URL / Sample)", "📤 Upload Side-View Video"],
            key="video_source_dropdown"
        )

    sample_video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "common_sample.mp4")

    if video_source_type == "🎥 YouTube Video (URL / Sample)":
        with ctrl_col2:
            yt_url = st.text_input(
                "YouTube Link (Editable):",
                value="https://youtube.com/shorts/TRvg083BrXY",
                help="Paste any YouTube video or short link here",
                key="yt_url_input"
            )
        default_url = "https://youtube.com/shorts/TRvg083BrXY"
        is_default = (yt_url.strip() == default_url or "TRvg083BrXY" in yt_url)

        if is_default and os.path.exists(sample_video_path):
            selected_video_path = sample_video_path
        else:
            if "custom_yt_path" in st.session_state and os.path.exists(st.session_state["custom_yt_path"]):
                selected_video_path = st.session_state["custom_yt_path"]

        with ctrl_col3:
            st.write("") # Spacer
            if not is_default and selected_video_path is None:
                if st.button("⬇️ Download Video", use_container_width=True):
                    with st.spinner("Downloading with yt-dlp..."):
                        os.makedirs("samples", exist_ok=True)
                        custom_path = os.path.join("samples", "custom_yt_input.mp4")
                        if os.path.exists(custom_path):
                            os.remove(custom_path)
                        res = os.system(f"./venv/bin/yt-dlp -f mp4 -o '{custom_path}' '{yt_url}'")
                        if res == 0 and os.path.exists(custom_path):
                            st.session_state["custom_yt_path"] = custom_path
                            st.success("Downloaded!")
                            st.rerun()
                        else:
                            st.error("Failed to download link.")
            else:
                run_analysis = st.button("🚀 Run AI Assessment", type="primary", use_container_width=True)

    else:
        with ctrl_col2:
            uploaded = st.file_uploader(
                "Upload Side-View Squat Video (MP4/MOV/AVI)",
                type=["mp4", "mov", "avi"],
                key="file_uploader_control"
            )
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read())
                tfile.close()
                selected_video_path = tfile.name
        with ctrl_col3:
            st.write("") # Spacer
            run_analysis = st.button("🚀 Run AI Assessment", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
st.divider()


# ── Main Content ──────────────────────────────────────────────────────────────
if selected_video_path:
    need_run = (
        run_analysis
        or "analysis_result" not in st.session_state
        or st.session_state.get("active_video") != selected_video_path
    )

    if run_analysis and need_run:
        prog = st.progress(0, text="Initialising Vision & Skill Engine…")
        stat = st.empty()

        def cb(pct, msg):
            prog.progress(int(pct * 100), text=msg)
            stat.caption(f"⚙️ {msg}")

        try:
            processor = SquatVideoProcessor()
            result = processor.process_video(
                selected_video_path, output_dir="output", progress_callback=cb
            )
            st.session_state["analysis_result"] = result
            st.session_state["active_video"]    = selected_video_path
            prog.empty(); stat.empty()
            st.toast("🎉 Analysis complete!", icon="✅")
        except Exception as exc:
            prog.empty(); stat.empty()
            st.error(f"Error: {exc}")

    result = st.session_state.get("analysis_result")

    if result:
        reps = result.get("repetitions", [])

        # ── KPI Row ──────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⏱ Duration",      f"{result['duration_sec']:.1f} s")
        c2.metric("🔁 Reps Detected", str(len(reps)))
        total_ev = sum(len(r["findings"]) for r in reps)
        passes   = sum(
            sum(1 for f in r["findings"] if f["status"] == "MEETS_STANDARD")
            for r in reps
        )
        c3.metric("✅ Pass Rate",  f"{passes / total_ev * 100:.0f}%" if total_ev else "—")
        c4.metric("🎬 Output",    "H.264 MP4")
        st.divider()

        # ── Tabs ─────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎥 Synced Video Player",
            "📋 Rep-by-Rep Audit",
            "📊 Biomechanics Telemetry",
            "📖 Skill Rule Inspector",
            "📷 Live Camera Counter",
        ])

        # TAB 1: Synced Dual Player
        with tab1:
            ann_path = result.get("annotated_video_path")
            if ann_path and os.path.exists(ann_path):
                synced_video_player(selected_video_path, ann_path)
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("📹 Original")
                    st.video(selected_video_path)
                with col_b:
                    st.warning("Annotated video not found.")

        # TAB 2: Rep-by-Rep Audit
        with tab2:
            if not reps:
                st.warning("No squat repetitions detected in the video.")
            else:
                rep_labels = [f"Repetition {r['rep_number']}" for r in reps]
                sel_rep    = st.selectbox("Select Rep:", rep_labels)
                rep_data   = reps[rep_labels.index(sel_rep)]

                st.markdown(f"### 🔍 Audit — Rep {rep_data['rep_number']}")
                st.caption(
                    f"Bottom @ **{rep_data['bottom_timestamp']:.2f} s** &nbsp;|&nbsp; "
                    f"Total **{rep_data['duration_sec']:.2f} s** "
                    f"(↓ {rep_data['descent_duration_sec']:.1f}s, ↑ {rep_data['ascent_duration_sec']:.1f}s)"
                )

                for f in rep_data.get("findings", []):
                    st_map = {
                        "MEETS_STANDARD":         ("status-badge-pass",   "✅ MEETS STANDARD"),
                        "DOES_NOT_MEET_STANDARD":  ("status-badge-fail",   "❌ DOES NOT MEET STANDARD"),
                        "CANNOT_ASSESS":           ("status-badge-cannot", "⚪ CANNOT ASSESS"),
                    }
                    badge_cls, badge_lbl = st_map.get(f["status"], ("status-badge-cannot", f["status"]))
                    st.markdown(f"""
                    <div class="card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <h4 style="margin:0;color:#f1f5f9;">{f['name']}</h4>
                        <span class="{badge_cls}">{badge_lbl}</span>
                      </div>
                      <p style="margin:4px 0;color:#94a3b8;font-size:.88rem;">
                        <strong>Observed:</strong> {f['observed']}
                      </p>
                      <p style="margin:4px 0;color:#cbd5e1;">{f['explanation']}</p>
                      <p style="margin:8px 0 0;color:#38bdf8;font-size:.86rem;">
                        💡 <strong>Feedback:</strong> {f['actionable_feedback']}
                      </p>
                      <div style="margin-top:10px;">
                        <span class="citation-pill">
                          📖 {f['citation']['ref_pages']} ({f['citation']['pdf_pages']}) · {f['citation']['section']}
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # TAB 3: Telemetry
        with tab3:
            st.subheader("Biomechanical Trajectory Plots")
            tel = result.get("frames_telemetry", [])
            if tel:
                df = pd.DataFrame([
                    {
                        "Timestamp (s)":           tf.get("timestamp", 0),
                        "Knee Angle (°)":          tf.get("knee_angle", 0),
                        "Back Angle (°)":          tf.get("back_angle", 0),
                        "Bar Midfoot Offset (px)": tf.get("bar_dev_px", 0),
                    }
                    for tf in tel if tf.get("valid", False)
                ])
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.caption("**Joint & Torso Angles over Time**")
                    st.line_chart(df.set_index("Timestamp (s)")[["Knee Angle (°)", "Back Angle (°)"]])
                with cc2:
                    st.caption("**Barbell Horizontal Offset from Midfoot**")
                    st.line_chart(df.set_index("Timestamp (s)")[["Bar Midfoot Offset (px)"]])

        # TAB 4: Skill Inspector
        with tab4:
            st.subheader("📖 Document-Derived Skill Rule Base")
            st.caption("Rules extracted from **Document_for_skill.pdf** — each criterion includes a page citation.")
            engine = SquatSkillEngine()
            for rule_id, rule in engine.rules.items():
                icon = "✅" if rule["assessable_from_side_view"] else "⚪"
                with st.expander(f"{icon} {rule['name']}  ·  {rule['category']}"):
                    st.markdown(f"**Description:** {rule['description']}")
                    side = "✅ Yes" if rule["assessable_from_side_view"] else "⚪ No (requires front/top camera)"
                    st.markdown(f"**Assessable from Side-View:** {side}")
                    st.markdown(
                        f"**Citation:** {rule['citation']['ref_pages']} "
                        f"({rule['citation']['pdf_pages']}) · {rule['citation']['section']}"
                    )
                    col_p, col_f = st.columns(2)
                    with col_p:
                        st.success(f"✅ Pass: {rule['pass_message']}")
                    with col_f:
                        st.error(f"❌ Fail: {rule['fail_message']}")

        # TAB 5: Live Camera Counter
        with tab5:
            _render_live_camera()

else:
    # No video loaded yet
    if app_mode == "📷 Live Camera Counter":
        _render_live_camera()
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#475569;">
          <div style="font-size:3.5rem;margin-bottom:16px;">🏋️</div>
          <h3 style="color:#64748b;font-family:'Outfit',sans-serif;">
            Select a video source in the sidebar to begin analysis
          </h3>
          <p style="color:#334155;font-size:.9rem;">
            Use the <strong>Common YouTube Sample</strong> or upload your own side-view squat video,
            then click <strong>🚀 Run AI Squat Assessment</strong>.<br/><br/>
            Or switch to <strong>📷 Live Camera Counter</strong> in the sidebar for real-time rep counting
            using your laptop camera — no video file needed.
          </p>
        </div>
        """, unsafe_allow_html=True)
