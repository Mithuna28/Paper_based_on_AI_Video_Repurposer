import re
import json
from pathlib import Path
from typing import List, Dict, Any

def escape_drawtext_text(text: str) -> str:
    """Escapes special characters for FFmpeg drawtext filter text parameter."""
    text = text.replace('\\', '\\\\')
    text = text.replace("'", "'\\''")
    text = text.replace(':', '\\:')
    text = text.replace('%', '\\%')
    return text

class ContentAnalyzer:
    """
    Content-Aware Animation & Visual Effect Engine.
    Analyzes transcript text and selected timestamps to generate dynamic,
    content-driven FFmpeg visual effects placed safely in the top overlay zone (y=140 to y=240).
    """

    def __init__(self, topic: str, font_path: str = "C\\:/Windows/Fonts/arialbd.ttf"):
        self.topic = topic
        self.font_path = font_path

    def analyze_segments(self, selected_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        effects = []

        # 1. Build a short topic-specific hook for the opening.
        topic_words = self.topic.split()
        if (
            len(topic_words) >= 3
            and topic_words[-1].lower() in {"basics", "fundamentals"}
            and topic_words[-2].lower() in {"programming", "coding"}
        ):
            del topic_words[-2]

        hook_subject = " ".join(topic_words).upper()
        hook_text = f"{hook_subject} YOU SHOULD KNOW"
        if len(hook_text) > 36:
            hook_text = hook_subject
        if len(hook_text) > 36:
            hook_text = f"{hook_text[:33].rsplit(' ', 1)[0]}..."

        effects.append({
            "type": "HOOK",
            "start": 0.0,
            "end": 3.5,
            "text": hook_text,
            "color": "white",
            "bg_color": "0x0F172A@0.85", # Dark slate card
            "position": "top_hook"
        })

        # Track timeline across concatenated clips
        current_time = 0.0
        step_counter = 1

        topic_words = set(re.findall(r'\w+', self.topic.lower()))

        for seg in selected_segments:
            seg_start = current_time
            seg_end = current_time + (seg["end"] - seg["start"])
            text = seg.get("text", "")
            lower_text = text.lower()

            # Rule A: Warning / Problem / Bug Callout
            if any(w in lower_text for w in ["error", "problem", "bug", "wrong", "fail", "careful", "mistake", "exception"]):
                effects.append({
                    "type": "WARNING",
                    "start": seg_start + 0.2,
                    "end": min(seg_end, seg_start + 3.2),
                    "text": "WARNING: COMMON ERROR",
                    "color": "yellow",
                    "bg_color": "0x991B1B@0.85", # Red card
                    "position": "top_card"
                })

            # Rule B: Step / Sequence Indicator
            elif any(w in lower_text for w in ["first", "second", "third", "step", "1.", "2.", "3.", "then", "next"]):
                effects.append({
                    "type": "STEP",
                    "start": seg_start + 0.2,
                    "end": min(seg_end, seg_start + 3.0),
                    "text": f"STEP {step_counter}: KEY CONCEPT",
                    "color": "white",
                    "bg_color": "0x1D4ED8@0.85", # Royal blue card
                    "position": "top_card"
                })
                step_counter += 1

            # Rule C: Tip / Shortcut Callout
            elif any(w in lower_text for w in ["tip", "shortcut", "trick", "best practice", "recommend", "easy"]):
                effects.append({
                    "type": "TIP",
                    "start": seg_start + 0.2,
                    "end": min(seg_end, seg_start + 3.0),
                    "text": "PRO TIP",
                    "color": "white",
                    "bg_color": "0x047857@0.85", # Emerald green card
                    "position": "top_card"
                })

            # Rule D: Example Callout
            elif any(w in lower_text for w in ["example", "sample", "instance", "here's", "call it"]):
                effects.append({
                    "type": "EXAMPLE",
                    "start": seg_start + 0.2,
                    "end": min(seg_end, seg_start + 3.0),
                    "text": "CODE EXAMPLE",
                    "color": "white",
                    "bg_color": "0x6D28D9@0.85", # Purple card
                    "position": "top_card"
                })

            # Rule E: Definition Callout
            elif any(w in lower_text for w in ["definition", "defined", "means", "what is", "refers to"]):
                effects.append({
                    "type": "DEFINITION",
                    "start": seg_start + 0.2,
                    "end": min(seg_end, seg_start + 3.2),
                    "text": "KEY DEFINITION",
                    "color": "white",
                    "bg_color": "0x4338CA@0.85", # Indigo card
                    "position": "top_card"
                })

            # Rule F: Keyword Pop for topic terms
            else:
                matched = [w for w in re.findall(r'\w+', lower_text) if w in topic_words and len(w) > 3]
                if matched:
                    kw = matched[0].upper()
                    effects.append({
                        "type": "KEYWORD",
                        "start": seg_start + 0.3,
                        "end": min(seg_end, seg_start + 2.5),
                        "text": f"TOPIC: {kw}",
                        "color": "yellow",
                        "bg_color": "0x0F172A@0.85",
                        "position": "top_card"
                    })

            current_time = seg_end

        return effects

    def build_ffmpeg_filter_chain(self, effects: List[Dict[str, Any]]) -> str:
        """Converts analyzed effect list into FFmpeg filter chain string for top overlay zone."""
        filter_parts = []

        for eff in effects:
            start = eff["start"]
            end = eff["end"]
            raw_text = eff["text"]
            escaped_text = escape_drawtext_text(raw_text)

            bg_col = eff.get("bg_color", "0x0F172A@0.85")
            text_col = eff.get("color", "white")
            pos = eff.get("position", "top_hook")

            if pos == "top_hook":
                y_expr = "140"
                x_expr = "(w-text_w)/2"
                fontsize = 32
                boxborder = 12
            else:
                y_expr = "220"
                x_expr = "(w-text_w)/2"
                fontsize = 28
                boxborder = 10

            filter_parts.append(
                f"drawtext=text='{escaped_text}':fontfile='{self.font_path}':"
                f"fontcolor={text_col}:fontsize={fontsize}:box=1:boxcolor={bg_col}:"
                f"boxborderw={boxborder}:x={x_expr}:y={y_expr}:"
                f"enable='between(t,{start:.2f},{end:.2f})'"
            )

        return ",".join(filter_parts)
