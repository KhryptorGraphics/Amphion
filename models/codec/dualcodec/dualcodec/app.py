# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gradio as gr
import torch
import torchaudio
import dualcodec
import base64
import soundfile as sf
import io

# Model configuration
MODEL_CONFIGS = {"12hz_v1": {"max_quantizers": 8}, "25hz_v1": {"max_quantizers": 12}}

w2v_path = "./w2v-bert-2.0"
dualcodec_model_path = "./dualcodec_ckpts"

# Global model variables
current_model = None
current_inference = None


def load_model(model_id):
    global current_model, current_inference
    current_model = dualcodec.get_model(model_id, dualcodec_model_path)
    current_inference = dualcodec.Inference(
        dualcodec_model=current_model,
        dualcodec_path=dualcodec_model_path,
        w2v_path=w2v_path,
        device="cuda",
    )
    return MODEL_CONFIGS[model_id]["max_quantizers"]


def toggle_theme(session_state):
    """Toggle between light and dark themes."""
    if session_state is None:
        session_state = {"history": [], "theme": "light"}

    # Toggle theme
    current_theme = session_state.get("theme", "light")
    new_theme = "dark" if current_theme == "light" else "light"
    session_state["theme"] = new_theme

    # Update theme button text
    button_text = "☀️ Light Mode" if new_theme == "dark" else "🌙 Dark Mode"

    # Return dynamic CSS for injection
    css_html = f"<style>{get_theme_css(new_theme)}</style>"

    return session_state, button_text, css_html


def get_theme_css(theme):
    """Return CSS styling based on theme."""
    if theme == "dark":
        return """
        body { background-color: #1a1a1a; color: #ffffff; }
        .gradio-container { background-color: #2d2d2d; color: #ffffff; }
        .panel { background-color: #3d3d3d; color: #ffffff; border: 1px solid #555555; }
        .label { color: #ffffff; }
        h1, h2, h3, h4, h5, h6 { color: #ffffff; }
        """
    else:
        return """
        body { background-color: #ffffff; color: #000000; }
        .gradio-container { background-color: #f0f0f0; color: #000000; }
        .panel { background-color: #ffffff; color: #000000; border: 1px solid #cccccc; }
        .label { color: #000000; }
        h1, h2, h3, h4, h5, h6 { color: #000000; }
        """


def process_audio(audio_file, model_id, n_quantizers, session_state):
    global current_model, current_inference
    if current_model is None or current_inference is None:
        load_model(model_id)

    # Load and process audio
    audio, sr = torchaudio.load(audio_file)
    audio = torchaudio.functional.resample(audio, sr, 24000)
    audio = audio.reshape(1, 1, -1)

    # Encode and decode
    semantic_codes, acoustic_codes = current_inference.encode(
        audio, n_quantizers=n_quantizers
    )
    out_audio = current_model.decode_from_codes(semantic_codes, acoustic_codes)

    # Prepare outputs
    generated_audio = (24000, out_audio.cpu().numpy().squeeze())

    # Update session state
    if session_state is None:
        session_state = {"history": [], "theme": "light"}

    # Ensure theme key exists
    if "theme" not in session_state:
        session_state["theme"] = "light"

    # Add new entry to history
    new_entry = {
        "audio": generated_audio,
        "metadata": f"Model: {model_id}, VQs: {n_quantizers}",
    }
    session_state["history"].append(new_entry)

    # Limit history to 10 entries
    if len(session_state["history"]) > 10:
        session_state["history"].pop(0)  # Remove the oldest entry

    return generated_audio, session_state


def update_slider(model_id):
    return gr.update(maximum=MODEL_CONFIGS[model_id]["max_quantizers"])


def generate_history_html(session_state):
    if session_state is None or "history" not in session_state:
        return ""
    history_list = session_state["history"]
    if not history_list:
        return ""
    html = []
    for idx, entry in enumerate(history_list):
        sr, audio_data = entry["audio"]
        # Convert numpy array to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format="wav")
        buffer.seek(0)
        data_uri = "data:audio/wav;base64," + base64.b64encode(buffer.read()).decode()
        html.append(
            f'<div style="border: 1px solid var(--border-color-primary); padding: 10px; margin: 10px; border-radius: 8px;">'
            f"<h4>History Entry {idx+1}</h4>"
            f'<audio controls><source src="{data_uri}" type="audio/wav"></audio>'
            f'<p>{entry["metadata"]}</p>'
            f"</div>"
        )
    return "".join(html)


def clear_history(session_state):
    if session_state is not None and "history" in session_state:
        session_state["history"] = []
    return session_state, ""


# Gradio interface
with gr.Blocks(css=get_theme_css("light")) as demo:
    gr.Markdown("# DualCodec Audio Demo")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()), value="12hz_v1", label="Model"
        )
        n_quantizers = gr.Slider(
            minimum=1,
            maximum=MODEL_CONFIGS["12hz_v1"]["max_quantizers"],
            step=1,
            value=8,
            label="Number of Quantizers",
        )
        theme_toggle = gr.Button("🌙 Dark Mode")

    audio_input = gr.Audio(type="filepath", label="Input Audio")
    inference_button = gr.Button("Run Inference")

    # Reconstructed audio output
    audio_output_recon = gr.Audio(label="Reconstructed Audio")

    # History section
    gr.Markdown("## History Outputs")
    history_display = gr.HTML(label="History Audios")

    # Session state to store history audios (unique to each user)
    session_state = gr.State({"history": [], "theme": "light"})

    # Dynamic CSS injection for theme switching
    theme_css_html = gr.HTML(
        value=f"<style>{get_theme_css('light')}</style>", visible=False
    )

    # Set up interactions
    model_dropdown.change(fn=update_slider, inputs=model_dropdown, outputs=n_quantizers)
    inference_button.click(
        fn=process_audio,
        inputs=[audio_input, model_dropdown, n_quantizers, session_state],
        outputs=[audio_output_recon, session_state],
    )
    session_state.change(
        fn=generate_history_html, inputs=session_state, outputs=history_display
    )

    # Theme toggle button
    theme_toggle.click(
        fn=toggle_theme,
        inputs=session_state,
        outputs=[session_state, theme_toggle, theme_css_html],
    )

    # Clear history button
    clear_button = gr.Button("Clear History Audios")
    clear_button.click(
        fn=clear_history, inputs=session_state, outputs=[session_state, history_display]
    )


def main():
    demo.launch()


if __name__ == "__main__":
    main()
