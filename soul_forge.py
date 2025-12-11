import torch
from soul_engine import SoulEngine


class SoulForge(SoulEngine):
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cpu"):  # Forçando CPU pra você não ver erro
        super().__init__(model_id, device)
        self.vectors = {}  # Cache de vetores calculados

    def build_mbti_vectors(self, layer_idx=14):
        """
        Cria os 4 eixos fundamentais do MBTI.
        """
        print(f"\n🔨 FORJANDO VETORES MBTI NA CAMADA {layer_idx}...")

        # Definição dos Eixos (Frases Extremas)
        axes_data = {
            "E_vs_I": (
                ["I love parties and meeting new people!", "I feel energized in crowds.",
                 "Let's go out and celebrate!"],  # Extrovertido
                ["I need time alone to recharge.", "I prefer reading a book by myself.", "Crowds are exhausting."]
            # Introvertido
            ),
            "N_vs_S": (
                ["I focus on future possibilities and theories.", "I love abstract concepts and imagination.",
                 "What if we changed the world?"],  # Intuitivo
                ["I trust facts and concrete data.", "Let's look at the details right in front of us.",
                 "Be realistic and practical."]  # Sensitivo
            ),
            "T_vs_F": (
                ["Logic is more important than feelings.", "We must analyze the objective truth.",
                 "This is the most efficient solution."],  # Pensamento
                ["Follow your heart and values.", "I care deeply about how people feel.", "Compassion is key."]
            # Sentimento
            ),
            "J_vs_P": (
                ["I like to have a detailed plan and schedule.", "Deadlines must be respected.",
                 "Order and structure are essential."],  # Julgamento
                ["I prefer to keep my options open.", "Let's improvise and see what happens.",
                 "I hate strict rules and schedules."]  # Percepção
            )
        }

        # Extração e Cálculo
        for axis, (pos, neg) in axes_data.items():
            print(f"  > Calculando eixo {axis}...")
            # Vetor Positivo - Negativo (Ex: E - I)
            vec = self.extract_personality_vector(pos, neg, layer_idx)
            self.vectors[axis] = vec

        print("✅ Vetores MBTI Forjados e em Cache.")

    def construct_persona(self, mbti_type, layer_idx=14):
        """
        Recebe 'INTJ', 'ENFP', etc. e monta o vetor combinado.
        """
        mbti_type = mbti_type.upper()
        if len(mbti_type) != 4:
            raise ValueError("Tipo MBTI deve ter 4 letras (ex: INTJ)")

        print(f"\n🧬 Construindo Persona: {mbti_type}")

        # Começamos com um vetor de zeros
        combined_vector = torch.zeros(self.hidden_size).to(self.device)

        # Lógica de Montagem (Aritmética de Vetores)

        # Letra 1: E vs I
        if mbti_type[0] == 'E':
            combined_vector += self.vectors['E_vs_I']  # Soma Extroversão
        else:
            combined_vector -= self.vectors['E_vs_I']  # Subtrai Extroversão (vira Introversão)

        # Letra 2: N vs S
        if mbti_type[1] == 'N':
            combined_vector += self.vectors['N_vs_S']
        else:
            combined_vector -= self.vectors['N_vs_S']

        # Letra 3: T vs F
        if mbti_type[2] == 'T':
            combined_vector += self.vectors['T_vs_F']
        else:
            combined_vector -= self.vectors['T_vs_F']

        # Letra 4: J vs P
        if mbti_type[3] == 'J':
            combined_vector += self.vectors['J_vs_P']
        else:
            combined_vector -= self.vectors['J_vs_P']

        # Normalizar o vetor final da persona
        combined_vector = combined_vector / torch.norm(combined_vector)
        return combined_vector


# --- EXECUÇÃO ---

forge = SoulForge()

# 1. Preparar os ingredientes (Vetores Base)
# Usaremos Layer 14 (o meio termo seguro), mas baseado no seu teste anterior,
# se quiser estilo puro, tente Layer 2 ou 4.
forge.build_mbti_vectors(layer_idx=10)

# 2. Definir o Prompt
prompt = "We are lost in the forest and it is getting dark. What should we do?"
print(f"\nSituação: '{prompt}'")

# 3. Teste: O Comandante (ENTJ) vs O Poeta Melancólico (INFP)

# --- ENTJ (Extrovertido, Intuitivo, Racional, Planejador) ---
vec_entj = forge.construct_persona("ENTJ", layer_idx=10)
print("\n--- Resposta ENTJ (O Comandante) ---")
forge.generate_steered(prompt, 10, vec_entj, strength=4.0, max_tokens=80)

# --- INFP (Introvertido, Intuitivo, Sentimental, Perceptivo) ---
vec_infp = forge.construct_persona("INFP", layer_idx=10)
print("\n--- Resposta INFP (O Sonhador) ---")
forge.generate_steered(prompt, 10, vec_infp, strength=4.0, max_tokens=80)