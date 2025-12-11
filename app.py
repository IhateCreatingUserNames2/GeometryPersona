import gradio as gr
import torch
import os
import shutil
import difflib
import numpy as np
from soul_cards import SoulCards

# --- 1. ENGINE INITIALIZATION ---
print("⏳ Initializing Soul Engine (This may take a moment)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
engine = SoulCards(device=device)
print(f"✅ Soul Engine Ready on {device}!")

# Global State
current_card = None
current_card_path = None


# --- 2. CALCULATION LOGIC (SCANNERS) ---

def calculate_divergence(text1, text2):
    """Measures the difference between two texts (0.0 to 1.0)."""
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return 1.0 - ratio


def scan_logic_common(test_prompt, vector_provider_fn):
    """
    Shared scan logic to avoid code repetition.
    vector_provider_fn(layer) -> returns the vector for that layer.
    """
    # 1. Baseline
    inputs = engine.tokenizer(test_prompt, return_tensors="pt").to(engine.device)
    base_out = engine.model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7)
    base_text = engine.tokenizer.decode(base_out[0], skip_special_tokens=True)

    report = f"🔍 **Baseline:** ...{base_text[-50:]}\n\n"
    best_layer = 14
    best_score = -1

    # 2. Layer Loop
    for layer in range(2, engine.model.config.num_hidden_layers, 2):
        try:
            # Get vector (either extracting new or getting from card)
            vec = vector_provider_fn(layer)
            if vec is None: continue

            # Lock seed
            torch.manual_seed(42)

            # Generate with intervention
            steered_text = engine.generate_steered(test_prompt, layer, vec, strength=4.0, max_tokens=40, verbose=False)

            # Measure impact
            div = calculate_divergence(base_text, steered_text)

            # Penalize broken responses (too short)
            if len(steered_text) < len(base_text) * 0.5: div = 0

            # Check Sweet Spot
            marker = ""
            if 0.4 < div < 0.9 and div > best_score:
                best_score = div
                best_layer = layer
                marker = "⭐"

            report += f"Layer {layer:02d}: Div {div:.2f} {marker}\n"

        except Exception as e:
            report += f"Layer {layer}: Error {e}\n"

    report += f"\n🏆 **Best Layer:** {best_layer} (Score: {best_score:.2f})"
    return report, best_layer


def run_forging_scanner(pos_text, neg_text, test_prompt):
    """Scanner for Creation tab (Calculates new vectors for each layer)."""
    if not pos_text or not neg_text: return "❌ Fill in the samples!", 14

    pos_samples = [x.strip() for x in pos_text.split('\n') if x.strip()]
    neg_samples = [x.strip() for x in neg_text.split('\n') if x.strip()]

    yield "⏳ Scanning (Calculating vectors)...", 14

    def provider(layer):
        return engine.extract_personality_vector(pos_samples, neg_samples, layer)

    report, best = scan_logic_common(test_prompt, provider)
    yield report, best


def run_equipped_scanner(test_prompt):
    """Scanner for Grimoire tab (Uses fixed card vector across layers)."""
    global current_card
    if current_card is None:
        yield "❌ No card equipped to diagnose!", 0
        return

    yield f"⏳ Diagnosing '{current_card['name']}' across layers...", 0

    # Simple provider: always returns the same card vector
    def provider(layer):
        return current_card['vector']

    report, best = scan_logic_common(test_prompt, provider)
    yield report, best


# --- 3. UI LOGIC ---

def chat_response(message, history, strength, layer_override):
    global current_card
    if current_card is None:
        inputs = engine.tokenizer(message, return_tensors="pt").to(engine.device)
        out = engine.model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        return engine.tokenizer.decode(out[0], skip_special_tokens=True)

    layer = int(layer_override) if layer_override > 0 else current_card['layer']
    vector = current_card['vector']

    response = engine.generate_steered(message, layer, vector, strength=strength, max_tokens=150, verbose=False)
    if response.startswith(message):
        response = response[len(message):].strip()
    return response


def refresh_deck_ui():
    files = engine.list_deck()
    if not files: return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=files, value=files[0] if files else None)


def load_card_ui(filename):
    global current_card, current_card_path
    if not filename: return "No file.", "Neutral", None

    try:
        current_card = engine.summon_card(filename)
        current_card_path = os.path.join(engine.cards_dir, filename)
        status_msg = f"✅ Card Equipped: {current_card['name']}"
        desc_msg = f"Description: {current_card['description']}\nBase Layer (Origin): {current_card['layer']}"
        return status_msg, desc_msg, current_card_path
    except Exception as e:
        return f"Error: {str(e)}", "Error", None


def unequip_card_ui():
    global current_card, current_card_path
    current_card = None
    current_card_path = None
    return "⚪ Base Mode (No Soul)", "Pure Original Model", None


def upload_card_ui(file_obj):
    if file_obj is None: return "No file uploaded."
    filename = os.path.basename(file_obj.name)
    destination = os.path.join(engine.cards_dir, filename)
    shutil.copy(file_obj.name, destination)
    return f"💾 Card '{filename}' imported! Click Refresh."


def craft_custom_ui(name, desc, pos_text, neg_text, layer):
    if not name or not pos_text or not neg_text: return "❌ Fill in all fields!"
    pos_samples = [x.strip() for x in pos_text.split('\n') if x.strip()]
    neg_samples = [x.strip() for x in neg_text.split('\n') if x.strip()]
    try:
        engine.craft_card(name, desc, pos_samples, neg_samples, int(layer))
        return f"✨ Card '{name}' created! Refresh the Grimoire."
    except Exception as e:
        return f"Error: {str(e)}"


def craft_mbti_ui(mbti_type, layer):
    if len(mbti_type) != 4: return "❌ MBTI type must be 4 letters."
    try:
        engine.craft_mbti_card(mbti_type, int(layer))
        return f"🧠 Card '{mbti_type}' created! Refresh the Grimoire."
    except Exception as e:
        return f"Error: {str(e)}"


# --- 4. GRADIO LAYOUT ---

with gr.Blocks(theme=gr.themes.Soft(), title="Soul Engine Studio") as app:
    gr.Markdown("# 🧬 Soul Engine Studio")
    gr.Markdown("Geometric Personality Manipulation.")

    with gr.Row():
        # --- COLUMN 1: CHAT ---
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Personality Test")
            with gr.Group():
                status_box = gr.Textbox(label="Equipped Card", value="⚪ No Soul", interactive=False)
                desc_box = gr.Textbox(label="Details", lines=2, interactive=False)

            with gr.Row():
                strength_slider = gr.Slider(-10, 10, 4.0, step=0.5, label="Strength")
                # Override slider can now be updated by Grimoire Scanner
                layer_override = gr.Slider(0, 24, 0, step=1, label="Layer Override (0 = Card's Original)")

            chatbot = gr.ChatInterface(fn=chat_response, additional_inputs=[strength_slider, layer_override])

        # --- COLUMN 2: TOOLS ---
        with gr.Column(scale=1):
            with gr.Tabs():
                # TAB 1: GRIMOIRE
                with gr.Tab("📖 Grimoire"):
                    gr.Markdown("### 1. Manage Deck")
                    deck_dropdown = gr.Dropdown(label="Available Cards", interactive=True)
                    refresh_btn = gr.Button("🔄 Refresh List")

                    with gr.Row():
                        load_btn = gr.Button("🔮 Equip", variant="primary")
                        unequip_btn = gr.Button("⚪ Unequip")

                    gr.Markdown("### 2. Equipped Card Diagnosis")
                    gr.Markdown("Tests if this card's vector works better on another layer.")
                    diag_prompt = gr.Textbox(value="Hello, who are you?", label="Test Prompt")
                    diag_btn = gr.Button("🔬 Scan Equipped Card", variant="secondary")
                    diag_report = gr.Textbox(label="Report", lines=4)

                    gr.Markdown("### 3. Files")
                    with gr.Row():
                        file_upload = gr.File(label="Import (.soul)", file_types=[".soul"])
                        file_download = gr.File(label="Download Card", interactive=False)
                    upload_msg = gr.Label()

                    # Grimoire Events
                    refresh_btn.click(refresh_deck_ui, outputs=deck_dropdown)
                    load_btn.click(load_card_ui, inputs=deck_dropdown, outputs=[status_box, desc_box, file_download])
                    unequip_btn.click(unequip_card_ui, outputs=[status_box, desc_box, file_download])
                    file_upload.upload(upload_card_ui, inputs=file_upload, outputs=upload_msg)

                    # Diagnostic Scanner: Updates Report AND Chat's Override Slider
                    diag_btn.click(
                        run_equipped_scanner,
                        inputs=[diag_prompt],
                        outputs=[diag_report, layer_override]
                    )

                # TAB 2: CUSTOM FORGE
                with gr.Tab("⚒️ Custom Forge"):
                    gr.Markdown("### Define Concepts")
                    cust_name = gr.Textbox(label="Name")
                    cust_desc = gr.Textbox(label="Description")
                    pos_in = gr.Textbox(lines=2, placeholder="Desired concept...", label="Positive Samples")
                    neg_in = gr.Textbox(lines=2, placeholder="Opposite...", label="Negative Samples")

                    gr.Markdown("### Creation Scanner")
                    scan_prompt = gr.Textbox(value="Hello, who are you?", label="Test Prompt")
                    scan_btn = gr.Button("🔍 Scan Layers", variant="secondary")
                    scan_report = gr.Textbox(label="Report", lines=4)

                    gr.Markdown("### Forge")
                    cust_layer = gr.Slider(0, 24, 14, step=1, label="Target Layer")
                    craft_btn = gr.Button("🔨 Forge Card", variant="primary")
                    craft_msg = gr.Label()

                    scan_btn.click(run_forging_scanner, inputs=[pos_in, neg_in, scan_prompt],
                                   outputs=[scan_report, cust_layer])
                    craft_btn.click(craft_custom_ui, inputs=[cust_name, cust_desc, pos_in, neg_in, cust_layer],
                                    outputs=craft_msg)

                # TAB 3: MBTI FORGE
                with gr.Tab("🧠 MBTI Forge"):
                    mbti_in = gr.Textbox(label="Type (e.g., INTJ)")
                    mbti_layer = gr.Slider(0, 24, 14, step=1, label="Target Layer")
                    mbti_btn = gr.Button("Forge MBTI", variant="primary")
                    mbti_msg = gr.Label()
                    mbti_btn.click(craft_mbti_ui, inputs=[mbti_in, mbti_layer], outputs=mbti_msg)

    app.load(refresh_deck_ui, outputs=deck_dropdown)

if __name__ == "__main__":
    app.launch()