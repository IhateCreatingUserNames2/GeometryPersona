import gradio as gr
import torch
import os
import shutil
import difflib
import numpy as np
from soul_cards import SoulCards

# --- 1. INICIALIZAÇÃO DO MOTOR ---
print("⏳ Inicializando Soul Engine (Isso pode demorar um pouco)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Instancia o motor de Cartas
engine = SoulCards(device=device)
print(f"✅ Soul Engine Pronta em {device}!")

# Estado Global
current_card = None
current_card_path = None  # Para download


# --- 2. LÓGICA DO SCANNER (Integrada para economizar RAM) ---

def calculate_divergence(text1, text2):
    """Mede a diferença entre dois textos."""
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return 1.0 - ratio


def run_ui_scanner(pos_text, neg_text, test_prompt):
    """
    Roda o scanner usando o motor já carregado.
    Retorna: Relatório (Texto) e o Melhor Layer (Número) para atualizar o slider.
    """
    if not pos_text or not neg_text or not test_prompt:
        return "❌ Preencha as amostras e o prompt de teste!", 14

    # Prepara os dados
    pos_samples = [x.strip() for x in pos_text.split('\n') if x.strip()]
    neg_samples = [x.strip() for x in neg_text.split('\n') if x.strip()]

    yield "⏳ Escaneando camadas... (Isso pode levar alguns segundos)", 14

    # Baseline
    inputs = engine.tokenizer(test_prompt, return_tensors="pt").to(engine.device)
    base_out = engine.model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7)
    base_text = engine.tokenizer.decode(base_out[0], skip_special_tokens=True)

    report = f"📝 **Baseline:** ...{base_text[-50:]}\n\n"
    best_layer = 14
    best_score = -1

    # Scan Loop (Pula de 2 em 2)
    for layer in range(2, engine.model.config.num_hidden_layers, 2):
        try:
            vec = engine.extract_personality_vector(pos_samples, neg_samples, layer)

            # Trava seed para comparação justa
            torch.manual_seed(42)

            steered_text = engine.generate_steered(test_prompt, layer, vec, strength=4.0, max_tokens=40, verbose=False)
            div = calculate_divergence(base_text, steered_text)

            # Filtro de sanidade (muito curto = erro)
            if len(steered_text) < len(base_text) * 0.5: div = 0

            marker = ""
            # Lógica do Sweet Spot (0.4 a 0.8 de mudança)
            if 0.4 < div < 0.9 and div > best_score:
                best_score = div
                best_layer = layer
                marker = "⭐"

            report += f"Layer {layer:02d}: Div {div:.2f} {marker}\n"

        except Exception as e:
            report += f"Layer {layer}: Erro {e}\n"

    report += f"\n🏆 **Melhor Camada Recomendada:** {best_layer} (Score: {best_score:.2f})"
    yield report, best_layer


# --- 3. LÓGICA DA UI (CHAT & ARQUIVOS) ---

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
    if not filename: return "Nenhum arquivo.", "Neutro", None

    try:
        current_card = engine.summon_card(filename)
        current_card_path = os.path.join(engine.cards_dir, filename)

        status_msg = f"✅ Carta Equipada: {current_card['name']}"
        desc_msg = f"Descrição: {current_card['description']}\nCamada Base: {current_card['layer']}"

        # Retorna info E o arquivo para download
        return status_msg, desc_msg, current_card_path
    except Exception as e:
        return f"Erro: {str(e)}", "Erro", None


def upload_card_ui(file_obj):
    """Recebe um arquivo .soul e salva na pasta cards/."""
    if file_obj is None: return "Nenhum arquivo enviado."

    filename = os.path.basename(file_obj.name)
    destination = os.path.join(engine.cards_dir, filename)

    shutil.copy(file_obj.name, destination)
    return f"💾 Carta '{filename}' importada com sucesso! Clique em Atualizar Lista."


def craft_custom_ui(name, desc, pos_text, neg_text, layer):
    if not name or not pos_text or not neg_text: return "❌ Preencha todos os campos!"
    pos_samples = [x.strip() for x in pos_text.split('\n') if x.strip()]
    neg_samples = [x.strip() for x in neg_text.split('\n') if x.strip()]

    try:
        engine.craft_card(name, desc, pos_samples, neg_samples, int(layer))
        return f"✨ Carta '{name}' criada (Layer {layer})! Vá para 'Grimório' e atualize."
    except Exception as e:
        return f"Erro: {str(e)}"


def craft_mbti_ui(mbti_type, layer):
    if len(mbti_type) != 4: return "❌ Tipo MBTI deve ter 4 letras."
    try:
        engine.craft_mbti_card(mbti_type, int(layer))
        return f"🧠 Carta '{mbti_type}' criada (Layer {layer})! Vá para 'Grimório' e atualize."
    except Exception as e:
        return f"Erro: {str(e)}"


# --- 4. LAYOUT GRADIO ---

with gr.Blocks(theme=gr.themes.Soft(), title="Soul Engine Studio") as app:
    gr.Markdown("# 🧬 Soul Engine Studio")
    gr.Markdown("Scanner, Forja e Manipulação de Personalidade via Geometria Latente.")

    with gr.Row():
        # --- COLUNA 1: CHAT ---
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Teste de Personalidade")

            with gr.Group():
                status_box = gr.Textbox(label="Carta Equipada", value="⚪ Sem Alma", interactive=False)
                desc_box = gr.Textbox(label="Detalhes", lines=2, interactive=False)

            with gr.Row():
                strength_slider = gr.Slider(-10, 10, 4.0, step=0.5, label="Força (Strength)")
                layer_override = gr.Slider(0, 24, 0, step=1, label="Layer Override (0=Auto)")

            chatbot = gr.ChatInterface(fn=chat_response, additional_inputs=[strength_slider, layer_override])

        # --- COLUNA 2: FERRAMENTAS ---
        with gr.Column(scale=1):
            with gr.Tabs():
                # ABA GRIMÓRIO (LOAD/EXPORT)
                with gr.Tab("📖 Grimório"):
                    gr.Markdown("### Gerenciar Deck")
                    deck_dropdown = gr.Dropdown(label="Cartas Disponíveis", interactive=True)
                    refresh_btn = gr.Button("🔄 Atualizar")

                    with gr.Row():
                        load_btn = gr.Button("🔮 Equipar", variant="primary")
                        unequip_btn = gr.Button("⚪ Desequipar")

                    gr.Markdown("---")
                    gr.Markdown("### Importar / Exportar")
                    with gr.Row():
                        file_upload = gr.File(label="Importar (.soul)", file_types=[".soul"])
                        file_download = gr.File(label="Baixar Carta Atual", interactive=False)

                    upload_msg = gr.Label()

                    # Eventos
                    refresh_btn.click(refresh_deck_ui, outputs=deck_dropdown)
                    load_btn.click(load_card_ui, inputs=deck_dropdown, outputs=[status_box, desc_box, file_download])
                    unequip_btn.click(lambda: (None, "⚪ Sem Alma", "Original"),
                                      outputs=[current_card, status_box, desc_box])
                    file_upload.upload(upload_card_ui, inputs=file_upload, outputs=upload_msg)

                # ABA FORJA CUSTOM (COM SCANNER)
                with gr.Tab("⚒️ Forja Custom"):
                    gr.Markdown("### 1. Definir Conceitos")
                    cust_name = gr.Textbox(label="Nome")
                    cust_desc = gr.Textbox(label="Descrição")
                    pos_in = gr.Textbox(lines=3, placeholder="Frases do conceito desejado...",
                                        label="Amostras Positivas")
                    neg_in = gr.Textbox(lines=3, placeholder="Frases do oposto...", label="Amostras Negativas")

                    gr.Markdown("### 2. Scanner de Sweet Spot")
                    gr.Markdown("Descubra qual camada funciona melhor para este conceito.")
                    scan_prompt = gr.Textbox(value="Hello, who are you?", label="Prompt de Teste do Scanner")
                    scan_btn = gr.Button("🔍 Escanear Layers", variant="secondary")
                    scan_report = gr.Textbox(label="Relatório do Scan", lines=6)

                    gr.Markdown("### 3. Forjar")
                    # O Slider recebe o valor do Scanner automaticamente
                    cust_layer = gr.Slider(0, 24, 14, step=1, label="Camada Alvo (Layer)")
                    craft_btn = gr.Button("🔨 Forjar e Salvar Carta", variant="primary")
                    craft_msg = gr.Label()

                    # Eventos
                    scan_btn.click(run_ui_scanner, inputs=[pos_in, neg_in, scan_prompt],
                                   outputs=[scan_report, cust_layer])
                    craft_btn.click(craft_custom_ui, inputs=[cust_name, cust_desc, pos_in, neg_in, cust_layer],
                                    outputs=craft_msg)

                # ABA FORJA MBTI
                with gr.Tab("🧠 Forja MBTI"):
                    mbti_in = gr.Textbox(label="Tipo (ex: ENTJ)")
                    mbti_layer = gr.Slider(0, 24, 14, step=1, label="Camada Alvo")
                    mbti_btn = gr.Button("Forjar MBTI", variant="primary")
                    mbti_msg = gr.Label()

                    mbti_btn.click(craft_mbti_ui, inputs=[mbti_in, mbti_layer], outputs=mbti_msg)

    app.load(refresh_deck_ui, outputs=deck_dropdown)

if __name__ == "__main__":
    app.launch()