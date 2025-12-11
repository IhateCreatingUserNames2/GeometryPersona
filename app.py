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
engine = SoulCards(device=device)
print(f"✅ Soul Engine Pronta em {device}!")

# Estado Global
current_card = None
current_card_path = None


# --- 2. LÓGICA DE CÁLCULO (SCANNERS) ---

def calculate_divergence(text1, text2):
    """Mede a diferença entre dois textos (0.0 a 1.0)."""
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return 1.0 - ratio


def scan_logic_common(test_prompt, vector_provider_fn):
    """
    Lógica compartilhada de scan para evitar repetição de código.
    vector_provider_fn(layer) -> devolve o vetor para aquela camada.
    """
    # 1. Baseline
    inputs = engine.tokenizer(test_prompt, return_tensors="pt").to(engine.device)
    base_out = engine.model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7)
    base_text = engine.tokenizer.decode(base_out[0], skip_special_tokens=True)

    report = f"📝 **Baseline:** ...{base_text[-50:]}\n\n"
    best_layer = 14
    best_score = -1

    # 2. Loop de Camadas
    for layer in range(2, engine.model.config.num_hidden_layers, 2):
        try:
            # Obtém o vetor (seja extraindo novo ou pegando o da carta)
            vec = vector_provider_fn(layer)
            if vec is None: continue

            # Trava seed
            torch.manual_seed(42)

            # Gera com intervenção
            steered_text = engine.generate_steered(test_prompt, layer, vec, strength=4.0, max_tokens=40, verbose=False)

            # Mede impacto
            div = calculate_divergence(base_text, steered_text)

            # Penaliza respostas quebradas (muito curtas)
            if len(steered_text) < len(base_text) * 0.5: div = 0

            # Verifica Sweet Spot
            marker = ""
            if 0.4 < div < 0.9 and div > best_score:
                best_score = div
                best_layer = layer
                marker = "⭐"

            report += f"Layer {layer:02d}: Div {div:.2f} {marker}\n"

        except Exception as e:
            report += f"Layer {layer}: Erro {e}\n"

    report += f"\n🏆 **Melhor Camada:** {best_layer} (Score: {best_score:.2f})"
    return report, best_layer


def run_forging_scanner(pos_text, neg_text, test_prompt):
    """Scanner da aba de Criação (Calcula vetores novos a cada camada)."""
    if not pos_text or not neg_text: return "❌ Preencha as amostras!", 14

    pos_samples = [x.strip() for x in pos_text.split('\n') if x.strip()]
    neg_samples = [x.strip() for x in neg_text.split('\n') if x.strip()]

    yield "⏳ Escaneando (Calculando vetores)...", 14

    def provider(layer):
        return engine.extract_personality_vector(pos_samples, neg_samples, layer)

    report, best = scan_logic_common(test_prompt, provider)
    yield report, best


def run_equipped_scanner(test_prompt):
    """Scanner da aba Grimório (Usa o vetor fixo da carta em várias camadas)."""
    global current_card
    if current_card is None:
        yield "❌ Nenhuma carta equipada para diagnosticar!", 0
        return

    yield f"⏳ Diagnosticando '{current_card['name']}' nas camadas...", 0

    # Provider simples: retorna sempre o mesmo vetor da carta
    def provider(layer):
        return current_card['vector']

    report, best = scan_logic_common(test_prompt, provider)
    yield report, best


# --- 3. LÓGICA DA UI ---

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
        desc_msg = f"Descrição: {current_card['description']}\nCamada Base (Origem): {current_card['layer']}"
        return status_msg, desc_msg, current_card_path
    except Exception as e:
        return f"Erro: {str(e)}", "Erro", None


def unequip_card_ui():
    global current_card, current_card_path
    current_card = None
    current_card_path = None
    return "⚪ Modo Base (Sem Alma)", "Modelo Original Puro", None


def upload_card_ui(file_obj):
    if file_obj is None: return "Nenhum arquivo enviado."
    filename = os.path.basename(file_obj.name)
    destination = os.path.join(engine.cards_dir, filename)
    shutil.copy(file_obj.name, destination)
    return f"💾 Carta '{filename}' importada! Clique em Atualizar."


def craft_custom_ui(name, desc, pos_text, neg_text, layer):
    if not name or not pos_text or not neg_text: return "❌ Preencha todos os campos!"
    pos_samples = [x.strip() for x in pos_text.split('\n') if x.strip()]
    neg_samples = [x.strip() for x in neg_text.split('\n') if x.strip()]
    try:
        engine.craft_card(name, desc, pos_samples, neg_samples, int(layer))
        return f"✨ Carta '{name}' criada! Atualize o Grimório."
    except Exception as e:
        return f"Erro: {str(e)}"


def craft_mbti_ui(mbti_type, layer):
    if len(mbti_type) != 4: return "❌ Tipo MBTI deve ter 4 letras."
    try:
        engine.craft_mbti_card(mbti_type, int(layer))
        return f"🧠 Carta '{mbti_type}' criada! Atualize o Grimório."
    except Exception as e:
        return f"Erro: {str(e)}"


# --- 4. LAYOUT GRADIO ---

with gr.Blocks(theme=gr.themes.Soft(), title="Soul Engine Studio") as app:
    gr.Markdown("# 🧬 Soul Engine Studio")
    gr.Markdown("Manipulação Geométrica de Personalidade.")

    with gr.Row():
        # --- COLUNA 1: CHAT ---
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Teste de Personalidade")
            with gr.Group():
                status_box = gr.Textbox(label="Carta Equipada", value="⚪ Sem Alma", interactive=False)
                desc_box = gr.Textbox(label="Detalhes", lines=2, interactive=False)

            with gr.Row():
                strength_slider = gr.Slider(-10, 10, 4.0, step=0.5, label="Força (Strength)")
                # Slider de Override agora pode ser atualizado pelo Scanner do Grimório
                layer_override = gr.Slider(0, 24, 0, step=1, label="Layer Override (0 = Original da Carta)")

            chatbot = gr.ChatInterface(fn=chat_response, additional_inputs=[strength_slider, layer_override])

        # --- COLUNA 2: FERRAMENTAS ---
        with gr.Column(scale=1):
            with gr.Tabs():
                # ABA 1: GRIMÓRIO
                with gr.Tab("📖 Grimório"):
                    gr.Markdown("### 1. Gerenciar Deck")
                    deck_dropdown = gr.Dropdown(label="Cartas Disponíveis", interactive=True)
                    refresh_btn = gr.Button("🔄 Atualizar Lista")

                    with gr.Row():
                        load_btn = gr.Button("🔮 Equipar", variant="primary")
                        unequip_btn = gr.Button("⚪ Desequipar")

                    gr.Markdown("### 2. Diagnóstico de Carta Equipada")
                    gr.Markdown("Testa se o vetor desta carta funciona melhor em outra camada.")
                    diag_prompt = gr.Textbox(value="Hello, who are you?", label="Prompt de Teste")
                    diag_btn = gr.Button("🔬 Escanear Carta Equipada", variant="secondary")
                    diag_report = gr.Textbox(label="Relatório", lines=4)

                    gr.Markdown("### 3. Arquivos")
                    with gr.Row():
                        file_upload = gr.File(label="Importar (.soul)", file_types=[".soul"])
                        file_download = gr.File(label="Baixar Carta", interactive=False)
                    upload_msg = gr.Label()

                    # Eventos Grimório
                    refresh_btn.click(refresh_deck_ui, outputs=deck_dropdown)
                    load_btn.click(load_card_ui, inputs=deck_dropdown, outputs=[status_box, desc_box, file_download])
                    unequip_btn.click(unequip_card_ui, outputs=[status_box, desc_box, file_download])
                    file_upload.upload(upload_card_ui, inputs=file_upload, outputs=upload_msg)

                    # Scanner de Diagnóstico: Atualiza o Relatório E o Slider de Override do Chat
                    diag_btn.click(
                        run_equipped_scanner,
                        inputs=[diag_prompt],
                        outputs=[diag_report, layer_override]
                    )

                # ABA 2: FORJA CUSTOM
                with gr.Tab("⚒️ Forja Custom"):
                    gr.Markdown("### Definir Conceitos")
                    cust_name = gr.Textbox(label="Nome")
                    cust_desc = gr.Textbox(label="Descrição")
                    pos_in = gr.Textbox(lines=2, placeholder="Conceito desejado...", label="Amostras Positivas")
                    neg_in = gr.Textbox(lines=2, placeholder="Oposto...", label="Amostras Negativas")

                    gr.Markdown("### Scanner de Criação")
                    scan_prompt = gr.Textbox(value="Hello, who are you?", label="Prompt de Teste")
                    scan_btn = gr.Button("🔍 Escanear Layers", variant="secondary")
                    scan_report = gr.Textbox(label="Relatório", lines=4)

                    gr.Markdown("### Forjar")
                    cust_layer = gr.Slider(0, 24, 14, step=1, label="Camada Alvo")
                    craft_btn = gr.Button("🔨 Forjar Carta", variant="primary")
                    craft_msg = gr.Label()

                    scan_btn.click(run_forging_scanner, inputs=[pos_in, neg_in, scan_prompt],
                                   outputs=[scan_report, cust_layer])
                    craft_btn.click(craft_custom_ui, inputs=[cust_name, cust_desc, pos_in, neg_in, cust_layer],
                                    outputs=craft_msg)

                # ABA 3: FORJA MBTI
                with gr.Tab("🧠 Forja MBTI"):
                    mbti_in = gr.Textbox(label="Tipo (ex: INTJ)")
                    mbti_layer = gr.Slider(0, 24, 14, step=1, label="Camada Alvo")
                    mbti_btn = gr.Button("Forjar MBTI", variant="primary")
                    mbti_msg = gr.Label()
                    mbti_btn.click(craft_mbti_ui, inputs=[mbti_in, mbti_layer], outputs=mbti_msg)

    app.load(refresh_deck_ui, outputs=deck_dropdown)

if __name__ == "__main__":
    app.launch()