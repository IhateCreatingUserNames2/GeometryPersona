import torch
from diffusers import StableDiffusionPipeline

# Configuração Otimizada para GTX 1060 6GB
device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5"

print(f"🎨 Carregando {model_id}...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # Importante: Usa metade da memória
    use_safetensors=True
)

# --- OTIMIZAÇÕES PARA 6GB VRAM ---
# Isso joga partes do modelo para a RAM normal quando não estão em uso.
# Deixa um pouco mais lento, mas garante que não vai travar (OOM).
pipe.enable_model_cpu_offload()
# pipe.enable_attention_slicing() # Descomente se ainda der erro de memória

# Função para extrair a "Alma" do texto
def get_embedding(text):
    tokens = pipe.tokenizer(text, return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids.to(device)
    with torch.no_grad():
        embedding = pipe.text_encoder(tokens)[0]
    return embedding

# --- EXPERIMENTO ---
positive_concept = "Cyberpunk city, neon lights"
negative_concept = "Forest, nature, organic"
prompt = "A portrait of a dog"

print("🧪 Extraindo vetores...")
emb_pos = get_embedding(positive_concept)
emb_neg = get_embedding(negative_concept)
emb_base = get_embedding(prompt)

# Aritmética
style_vector = emb_pos - emb_neg
strength = 0.6
steered_embedding = emb_base + (style_vector * strength)

print(f"🖌️ Gerando imagem...")

with torch.no_grad():
    image = pipe(
        prompt_embeds=steered_embedding,
        height=512,
        width=512,
        num_inference_steps=30 # Se diminuir para 20, fica mais rápido (10-15s)
    ).images[0]

image.save("dog_cyber_1060.png")
print("✅ Imagem salva!")