# coding=utf-8
# Qwen3-TTS Gradio Demo powered by vLLM Omni
# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice)
# Lazy loading - models are loaded on-demand when clicking "Load Model" button
# Both 0.6B and 1.7B models are available
import os
import tempfile

import gradio as gr
import numpy as np
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from end2end import _estimate_prompt_len

from vllm_omni import Omni

# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan",
    "Serena", "Sohee", "Uncle_fu", "Vivian",
]
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian",
    "Italian", "Dutch", "Turkish", "Arabic",
]
MODEL_SIZES = ["0.6B", "1.7B"]
# VoiceDesign only has a 1.7B variant
VOICE_DESIGN_SIZES = ["1.7B"]

# Low-memory stage config for single-GPU setups (0.2 + 0.15 = 35% GPU mem)
STAGE_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "stage_configs_lowmem.yaml")

# Global Omni instances (loaded lazily)
voice_design_omni: Omni | None = None
base_omni: Omni | None = None
custom_voice_omni: Omni | None = None


def _model_name(model_type: str, size: str) -> str:
    return f"Qwen/Qwen3-TTS-12Hz-{size}-{model_type}"


def _extract_audio(mm: dict) -> tuple[int, np.ndarray]:
    """Extract audio from multimodal output as (sample_rate, numpy_array)."""
    audio_data = mm["audio"]
    sr_raw = mm["sr"]
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
    audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    return sr, audio_tensor.float().cpu().numpy().flatten()


def _build_input(additional_information: dict, model_name: str) -> dict:
    """Build Omni input dict with estimated prompt length."""
    prompt_len = _estimate_prompt_len(additional_information, model_name)
    return {
        "prompt_token_ids": [0] * prompt_len,
        "additional_information": additional_information,
    }


def _generate_single(omni: Omni, inputs: dict) -> tuple[int, np.ndarray]:
    """Run Omni.generate for a single input and return (sr, audio_numpy)."""
    for stage_outputs in omni.generate([inputs]):
        for output in stage_outputs.request_output:
            mm = output.outputs[0].multimodal_output
            return _extract_audio(mm)
    raise RuntimeError("No output produced by Omni.generate")


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _save_ref_audio_to_temp(audio_tuple) -> str:
    """Save (wav, sr) tuple to a temporary wav file and return the path."""
    import soundfile as sf

    wav, sr = audio_tuple
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav, samplerate=sr, format="WAV")
    return tmp.name


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_voice_design_model(model_size, progress=gr.Progress(track_tqdm=True)):
    global voice_design_omni
    if voice_design_omni is not None:
        return "Model already loaded. Restart to switch size."
    try:
        progress(0, desc=f"Loading VoiceDesign {model_size} model...")
        model_name = _model_name("VoiceDesign", model_size)
        voice_design_omni = Omni(model=model_name, stage_configs_path=STAGE_CONFIGS_PATH)
        progress(1.0, desc="Model loaded successfully!")
        return f"VoiceDesign {model_size} model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


def load_base_model(model_size, progress=gr.Progress(track_tqdm=True)):
    global base_omni
    if base_omni is not None:
        return "Model already loaded. Restart to switch size."
    try:
        progress(0, desc=f"Loading Base {model_size} model...")
        model_name = _model_name("Base", model_size)
        base_omni = Omni(model=model_name, stage_configs_path=STAGE_CONFIGS_PATH)
        progress(1.0, desc="Model loaded successfully!")
        return f"Base {model_size} model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


def load_custom_voice_model(model_size, progress=gr.Progress(track_tqdm=True)):
    global custom_voice_omni
    if custom_voice_omni is not None:
        return "Model already loaded. Restart to switch size."
    try:
        progress(0, desc=f"Loading CustomVoice {model_size} model...")
        model_name = _model_name("CustomVoice", model_size)
        custom_voice_omni = Omni(model=model_name, stage_configs_path=STAGE_CONFIGS_PATH)
        progress(1.0, desc="Model loaded successfully!")
        return f"CustomVoice {model_size} model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {type(e).__name__}: {e}"


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_voice_design(text, language, voice_description, progress=gr.Progress(track_tqdm=True)):
    if voice_design_omni is None:
        return None, "Error: Model not loaded. Please click 'Load Model' first."
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        model_name = _model_name("VoiceDesign", "1.7B")  # name only used for estimation
        additional_information = {
            "task_type": ["VoiceDesign"],
            "text": [text.strip()],
            "language": [language],
            "instruct": [voice_description.strip()],
            "max_new_tokens": [2048],
            "non_streaming_mode": [True],
        }
        inputs = _build_input(additional_information, model_name)
        sr, audio = _generate_single(voice_design_omni, inputs)
        return (sr, audio), "Voice design generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, progress=gr.Progress(track_tqdm=True)):
    if base_omni is None:
        return None, "Error: Model not loaded. Please click 'Load Model' first."
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

    try:
        temp_path = _save_ref_audio_to_temp(audio_tuple)
        model_name = _model_name("Base", "1.7B")
        additional_information = {
            "task_type": ["Base"],
            "ref_audio": [temp_path],
            "ref_text": [ref_text.strip() if ref_text else ""],
            "text": [target_text.strip()],
            "language": [language],
            "x_vector_only_mode": [use_xvector_only],
            "max_new_tokens": [2048],
        }
        inputs = _build_input(additional_information, model_name)
        sr, audio = _generate_single(base_omni, inputs)
        os.unlink(temp_path)
        return (sr, audio), "Voice clone generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, progress=gr.Progress(track_tqdm=True)):
    if custom_voice_omni is None:
        return None, "Error: Model not loaded. Please click 'Load Model' first."
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    try:
        model_name = _model_name("CustomVoice", "1.7B")
        additional_information = {
            "task_type": ["CustomVoice"],
            "text": [text.strip()],
            "language": [language],
            "speaker": [speaker],
            "instruct": [instruct.strip() if instruct else ""],
            "max_new_tokens": [2048],
        }
        inputs = _build_input(additional_information, model_name)
        sr, audio = _generate_single(custom_voice_omni, inputs)
        return (sr, audio), "Generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


# ============================================================================
# GRADIO UI
# ============================================================================

def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo (vLLM Omni)") as demo:
        gr.Markdown(
            """
# Qwen3-TTS Demo (vLLM Omni)
A unified Text-to-Speech demo featuring three powerful modes:
- **Voice Design**: Create custom voices using natural language descriptions
- **Voice Clone (Base)**: Clone any voice from a reference audio
- **TTS (CustomVoice)**: Generate speech with predefined speakers and optional style instructions
Powered by [vLLM Omni](https://github.com/vllm-project/vllm) for accelerated inference.
"""
        )

        with gr.Tabs():
            # Tab 1: Voice Design
            with gr.Tab("Voice Design"):
                gr.Markdown("### Create Custom Voice with Natural Language")
                gr.Markdown("**Note:** Click 'Load Model' before generating audio.")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            design_size = gr.Dropdown(label="Model Size", choices=VOICE_DESIGN_SIZES, value="1.7B")
                            design_load_btn = gr.Button("Load Model", variant="secondary")
                        design_load_status = gr.Textbox(label="Load Status", lines=1, interactive=False)
                        design_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                            interactive=False,
                        )
                        design_language = gr.Dropdown(
                            label="Language", choices=LANGUAGES, value="Auto", interactive=False,
                        )
                        design_instruct = gr.Textbox(
                            label="Voice Description",
                            lines=3,
                            placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                            interactive=False,
                        )
                        design_btn = gr.Button("Generate with Custom Voice", variant="primary", interactive=False)

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)

                design_load_btn.click(
                    load_voice_design_model,
                    inputs=[design_size],
                    outputs=[design_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[design_text, design_language, design_instruct, design_btn],
                )

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio_out, design_status],
                )

            # Tab 2: Voice Clone (Base)
            with gr.Tab("Voice Clone (Base)"):
                gr.Markdown("### Clone Voice from Reference Audio")
                gr.Markdown("**Note:** Click 'Load Model' before generating audio.")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            clone_size = gr.Dropdown(label="Model Size", choices=MODEL_SIZES, value="1.7B")
                            clone_load_btn = gr.Button("Load Model", variant="secondary")
                        clone_load_status = gr.Textbox(label="Load Status", lines=1, interactive=False)
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio (Upload a voice sample to clone)",
                            type="numpy",
                            interactive=False,
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference Text (Transcript of the reference audio)",
                            lines=2,
                            placeholder="Enter the exact text spoken in the reference audio...",
                            interactive=False,
                        )
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only (No reference text needed, but lower quality)",
                            value=False,
                            interactive=False,
                        )

                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="Target Text (Text to synthesize with cloned voice)",
                            lines=4,
                            placeholder="Enter the text you want the cloned voice to speak...",
                            interactive=False,
                        )
                        clone_language = gr.Dropdown(
                            label="Language", choices=LANGUAGES, value="Auto", interactive=False,
                        )
                        clone_btn = gr.Button("Clone & Generate", variant="primary", interactive=False)

                with gr.Row():
                    clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

                clone_load_btn.click(
                    load_base_model,
                    inputs=[clone_size],
                    outputs=[clone_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[clone_ref_audio, clone_ref_text, clone_xvector, clone_target_text, clone_language, clone_btn],
                )

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector],
                    outputs=[clone_audio_out, clone_status],
                )

            # Tab 3: TTS (CustomVoice)
            with gr.Tab("TTS (CustomVoice)"):
                gr.Markdown("### Text-to-Speech with Predefined Speakers")
                gr.Markdown("**Note:** Click 'Load Model' before generating audio.")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            tts_size = gr.Dropdown(label="Model Size", choices=MODEL_SIZES, value="1.7B")
                            tts_load_btn = gr.Button("Load Model", variant="secondary")
                        tts_load_status = gr.Textbox(label="Load Status", lines=1, interactive=False)
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities.",
                            interactive=False,
                        )
                        tts_language = gr.Dropdown(
                            label="Language", choices=LANGUAGES, value="English", interactive=False,
                        )
                        tts_speaker = gr.Dropdown(
                            label="Speaker", choices=SPEAKERS, value="Ryan", interactive=False,
                        )
                        tts_instruct = gr.Textbox(
                            label="Style Instruction (Optional)",
                            lines=2,
                            placeholder="e.g., Speak in a cheerful and energetic tone",
                            interactive=False,
                        )
                        tts_btn = gr.Button("Generate Speech", variant="primary", interactive=False)

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

                tts_load_btn.click(
                    load_custom_voice_model,
                    inputs=[tts_size],
                    outputs=[tts_load_status],
                ).then(
                    lambda: (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    ),
                    outputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_btn],
                )

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct],
                    outputs=[tts_audio_out, tts_status],
                )

        gr.Markdown(
            """
---
**Note**: Models are loaded on-demand when you click the "Load Model" button.
For longer texts, please split them into smaller segments.
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
