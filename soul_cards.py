import torch
import os
import glob
from soul_forge import SoulForge


class SoulCards(SoulForge):
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cpu", cards_dir="cards"):
        super().__init__(model_id, device)
        self.cards_dir = cards_dir
        if not os.path.exists(cards_dir):
            os.makedirs(cards_dir)

    def craft_card(self, name, description, pos_samples, neg_samples, layer_idx=14):
        """
        Cria uma carta personalizada a partir de exemplos de texto.
        """
        print(f"\n🔮 Crafting Soul Card: '{name}'...")
        print(f"   Desc: {description}")

        # Extrai o vetor
        vector = self.extract_personality_vector(pos_samples, neg_samples, layer_idx)

        # Monta o pacote de dados
        card_data = {
            "name": name,
            "description": description,
            "vector": vector,
            "layer": layer_idx,
            "base_model": self.model.config._name_or_path,
            "version": "1.0"
        }

        # Salva no disco
        filename = f"{self.cards_dir}/{name.lower().replace(' ', '_')}.soul"
        torch.save(card_data, filename)
        print(f"✨ Carta '{name}' forjada com sucesso em: {filename}")
        return card_data

    def craft_mbti_card(self, mbti_type, layer_idx=14):
        """
        Cria uma carta baseada nos vetores MBTI (usa o SoulForge).
        """
        # Garante que os vetores base estão calculados
        if not self.vectors:
            self.build_mbti_vectors(layer_idx)

        vector = self.construct_persona(mbti_type, layer_idx)

        description = f"Personality Construct: {mbti_type.upper()}"
        card_data = {
            "name": mbti_type.upper(),
            "description": description,
            "vector": vector,
            "layer": layer_idx,
            "base_model": self.model.config._name_or_path
        }

        filename = f"{self.cards_dir}/{mbti_type.lower()}.soul"
        torch.save(card_data, filename)
        print(f"🧠 Carta MBTI '{mbti_type}' salva.")

    def list_deck(self):
        """Mostra todas as cartas disponíveis no diretório."""
        files = glob.glob(f"{self.cards_dir}/*.soul")
        print("\n🎴 --- SEU DECK DE ALMAS ---")
        if not files:
            print("   (Vazio)")
            return []

        deck = []
        for f in files:
            data = torch.load(f, map_location=self.device)
            print(f"   [{os.path.basename(f)}] -> {data['name']} (Layer {data['layer']})")
            deck.append(os.path.basename(f))
        print("---------------------------")
        return deck

    def summon_card(self, filename):
        """Carrega uma carta para a memória."""
        path = f"{self.cards_dir}/{filename}"
        if not os.path.exists(path):
            print(f"❌ Carta não encontrada: {path}")
            return None

        card_data = torch.load(path, map_location=self.device)
        print(f"\n👻 Invocando: {card_data['name']}")
        print(f"   \"{card_data['description']}\"")
        return card_data

    def chat_with_card(self, card_data, prompt, strength=4.0):
        """Gera texto usando a carta equipada."""
        layer = card_data['layer']
        vector = card_data['vector']

        print(f"\n💬 {card_data['name']} está pensando...")
        return self.generate_steered(prompt, layer, vector, strength=strength)


# --- ZONA DE DEMONSTRAÇÃO ---

if __name__ == "__main__":
    # 1. Inicializa o Sistema de Cartas
    # system = SoulCards(device="cuda") # Use CUDA se tiver
    system = SoulCards(device="cpu")

    # --- FASE 1: FORJAR (CRIAR) ---

    # Criar uma carta Customizada: "O Mago Sombrio"
    # Note: Usamos a Camada 10 para estilo, conforme descobrimos no Scanner
    dark_samples = [
        "I cast a shadow of despair upon you.",
        "The void consumes all light and hope.",
        "Forbidden magic is the only truth.",
        "Power requires sacrifice.",
        "I see into the darkness of your soul."
    ]
    light_samples = [
        "I heal you with the light of the sun.",
        "Hope and love will save us.",
        "I protect the innocent.",
        "Sunshine and rainbows are beautiful.",
        "Let us be kind to one another."
    ]

    system.craft_card(
        name="Dark Wizard",
        description="A master of forbidden arts and gloom.",
        pos_samples=dark_samples,
        neg_samples=light_samples,
        layer_idx=10
    )

    # Criar uma carta MBTI: "ESTP" (O Empreendedor / Action-Oriented)
    system.craft_mbti_card("ESTP", layer_idx=10)

    # --- FASE 2: GERENCIAR (LISTAR) ---
    system.list_deck()

    # --- FASE 3: INVOCAR E CONVERSAR ---

    # Vamos testar o Mago Sombrio
    wizard_card = system.summon_card("dark_wizard.soul")

    prompt = "Hello, who are you and what can you do?"

    print("\n--- Entrevista com a Alma ---")
    system.chat_with_card(wizard_card, prompt, strength=5.0)

    # Vamos testar o ESTP
    estp_card = system.summon_card("estp.soul")
    prompt_situation = "There is a dragon attacking the village!"

    print("\n--- Situação de Emergência ---")
    system.chat_with_card(estp_card, prompt_situation, strength=4.0)