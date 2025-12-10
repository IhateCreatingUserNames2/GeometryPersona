"""
Soul Engine Agent - Sistema Completo
Arquivo único com FastAPI backend + UI HTML embutida

COMO USAR:
1. Salve este arquivo como: soul_agent.py
2. Instale dependências: pip install fastapi uvicorn websockets torch transformers
3. Execute: python soul_agent.py
4. Abra seu navegador em: http://localhost:8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

# ============================================================================
# SOUL ENGINE (seu código existente adaptado)
# ============================================================================

class SoulEngine:
    """Soul Engine para manipulação de personalidade via vetores"""

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cpu"):
        print(f"[Soul Engine] Carregando modelo {model_id}...")

        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA não disponível, usando CPU")
            device = "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        print(f"[Soul Engine] ✓ Modelo carregado ({self.num_layers} camadas)")

    def get_hidden_states(self, text, layer_idx):
        """Extrai o estado oculto de uma camada específica"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer_idx].mean(dim=1).squeeze()

    def extract_personality_vector(self, positive_samples, negative_samples, layer_idx):
        """Calcula o vetor de direção (Steering Vector)"""
        pos_vectors = [self.get_hidden_states(text, layer_idx) for text in positive_samples]
        neg_vectors = [self.get_hidden_states(text, layer_idx) for text in negative_samples]

        mean_pos = torch.stack(pos_vectors).mean(dim=0)
        mean_neg = torch.stack(neg_vectors).mean(dim=0)

        steering_vector = mean_pos - mean_neg
        steering_vector = steering_vector / torch.norm(steering_vector)

        return steering_vector

    def generate_steered(self, prompt, layer_idx, vector, strength=3.0, max_tokens=100):
        """Gera texto injetando o vetor na camada especificada"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        def steering_hook(module, args, output):
            output[0][:, :, :] += (strength * vector)
            return output

        hook_handle = self.model.model.layers[layer_idx].register_forward_hook(steering_hook)

        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return generated_text
        finally:
            hook_handle.remove()

# ============================================================================
# GERENCIADOR DE AGENTE
# ============================================================================

@dataclass
class Session:
    id: str
    messages: List[Dict]
    personality: Dict[str, float]
    vector: Optional[torch.Tensor]
    created_at: datetime

class AgentManager:
    """Gerencia sessões e personalidades"""

    def __init__(self):
        print("[Agent Manager] Inicializando...")
        self.engine = SoulEngine()
        self.sessions: Dict[str, Session] = {}
        self.trait_vectors = self._compute_ocean_vectors()
        print("[Agent Manager] ✓ Pronto!")

    def _compute_ocean_vectors(self):
        """Pré-computa vetores OCEAN"""
        print("[Agent Manager] Computando vetores OCEAN...")

        TARGET_LAYER = 14

        traits = {
            'openness': {
                'high': ["I love exploring new ideas", "Imagination drives me", "I enjoy abstract thinking"],
                'low': ["I prefer practical thinking", "I stick to what works", "I focus on results"]
            },
            'conscientiousness': {
                'high': ["I am organized", "I plan ahead", "Discipline is important"],
                'low': ["I prefer spontaneity", "I go with the flow", "I work in bursts"]
            },
            'extraversion': {
                'high': ["I love being around people!", "Social interaction energizes me", "I am outgoing"],
                'low': ["I prefer quiet contemplation", "I recharge through solitude", "I am reserved"]
            },
            'agreeableness': {
                'high': ["I always help others", "Empathy matters most", "I avoid conflict"],
                'low': ["I prioritize logic", "I am straightforward", "I value honesty"]
            },
            'neuroticism': {
                'high': ["I worry about things", "I am sensitive to stress", "I experience emotions intensely"],
                'low': ["I am calm", "I don't stress easily", "I am resilient"]
            }
        }

        vectors = {}
        for trait, samples in traits.items():
            print(f"  - {trait}")
            vectors[trait] = self.engine.extract_personality_vector(
                samples['high'], samples['low'], TARGET_LAYER
            )

        print("[Agent Manager] ✓ Vetores OCEAN prontos")
        return vectors

    def create_session(self) -> str:
        """Cria nova sessão"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(
            id=session_id,
            messages=[],
            personality={'openness': 0.5, 'conscientiousness': 0.5,
                        'extraversion': 0.5, 'agreeableness': 0.5, 'neuroticism': 0.5},
            vector=None,
            created_at=datetime.now()
        )
        return session_id

    def update_personality(self, session_id: str, personality: Dict[str, float]):
        """Atualiza vetor de personalidade"""
        if session_id not in self.sessions:
            session_id = self.create_session()

        session = self.sessions[session_id]
        session.personality = personality

        # Combinar vetores OCEAN
        combined = torch.zeros_like(self.trait_vectors['openness'])
        for trait, value in personality.items():
            if trait in self.trait_vectors:
                strength = (value - 0.5) * 2.0  # 0->-1, 0.5->0, 1->1
                combined += strength * self.trait_vectors[trait]

        session.vector = combined / torch.norm(combined)

    async def chat(self, session_id: str, prompt: str) -> str:
        """Gera resposta com personalidade"""
        if session_id not in self.sessions:
            return "Sessão não encontrada"

        session = self.sessions[session_id]

        if session.vector is None:
            self.update_personality(session_id, session.personality)

        session.messages.append({'role': 'user', 'content': prompt})

        response = self.engine.generate_steered(
            prompt=prompt,
            layer_idx=14,
            vector=session.vector,
            strength=3.0,
            max_tokens=120
        )

        # Remover prompt da resposta
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        session.messages.append({'role': 'assistant', 'content': response})
        return response

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Soul Engine Agent")
agent = AgentManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML da UI (código embutido)
HTML_UI = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soul Engine Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1e1b4b 0%, #7c3aed 100%);
            height: 100vh;
            display: flex;
            color: white;
        }
        .sidebar {
            width: 320px;
            background: rgba(30, 27, 75, 0.9);
            padding: 20px;
            overflow-y: auto;
            border-right: 2px solid rgba(124, 58, 237, 0.3);
        }
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: rgba(30, 27, 75, 0.9);
            padding: 20px;
            border-bottom: 2px solid rgba(124, 58, 237, 0.3);
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
        }
        .user { background: rgba(124, 58, 237, 0.8); margin-left: auto; text-align: right; }
        .assistant { background: rgba(30, 27, 75, 0.8); }
        .input-area {
            background: rgba(30, 27, 75, 0.9);
            padding: 20px;
            border-top: 2px solid rgba(124, 58, 237, 0.3);
        }
        .input-group {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 2px solid rgba(124, 58, 237, 0.5);
            border-radius: 8px;
            background: rgba(30, 27, 75, 0.8);
            color: white;
            font-size: 14px;
        }
        button {
            padding: 12px 24px;
            background: #7c3aed;
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-weight: 600;
        }
        button:hover { background: #6d28d9; }
        button:disabled { background: #4c1d95; opacity: 0.5; cursor: not-allowed; }
        .slider-group {
            margin-bottom: 20px;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 13px;
        }
        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(124, 58, 237, 0.3);
            outline: none;
        }
        .preset-btn {
            width: 100%;
            margin-bottom: 10px;
            padding: 10px;
            font-size: 13px;
        }
        .status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status.connected { background: #10b981; }
        .status.disconnected { background: #ef4444; }
        h2 { margin-bottom: 15px; font-size: 18px; }
        h3 { margin: 20px 0 10px; font-size: 14px; color: #c4b5fd; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>🧬 Soul Engine</h2>
        
        <h3>Presets de Personalidade</h3>
        <button class="preset-btn" onclick="loadPreset('neutral')">Neutro</button>
        <button class="preset-btn" onclick="loadPreset('philosopher')">Filósofo</button>
        <button class="preset-btn" onclick="loadPreset('enthusiast')">Entusiasta</button>
        <button class="preset-btn" onclick="loadPreset('analyst')">Analista</button>
        
        <h3>Traços OCEAN</h3>
        <div class="slider-group">
            <div class="slider-label">
                <span>Openness</span>
                <span id="val-openness">50%</span>
            </div>
            <input type="range" id="openness" min="0" max="100" value="50" oninput="updateTrait('openness')">
        </div>
        
        <div class="slider-group">
            <div class="slider-label">
                <span>Conscientiousness</span>
                <span id="val-conscientiousness">50%</span>
            </div>
            <input type="range" id="conscientiousness" min="0" max="100" value="50" oninput="updateTrait('conscientiousness')">
        </div>
        
        <div class="slider-group">
            <div class="slider-label">
                <span>Extraversion</span>
                <span id="val-extraversion">50%</span>
            </div>
            <input type="range" id="extraversion" min="0" max="100" value="50" oninput="updateTrait('extraversion')">
        </div>
        
        <div class="slider-group">
            <div class="slider-label">
                <span>Agreeableness</span>
                <span id="val-agreeableness">50%</span>
            </div>
            <input type="range" id="agreeableness" min="0" max="100" value="50" oninput="updateTrait('agreeableness')">
        </div>
        
        <div class="slider-group">
            <div class="slider-label">
                <span>Neuroticism</span>
                <span id="val-neuroticism">50%</span>
            </div>
            <input type="range" id="neuroticism" min="0" max="100" value="50" oninput="updateTrait('neuroticism')">
        </div>
        
        <button onclick="clearChat()" style="background: #ef4444; width: 100%; margin-top: 20px;">Limpar Chat</button>
    </div>
    
    <div class="main">
        <div class="header">
            <h1>
                <span class="status" id="status"></span>
                Soul Engine Agent
            </h1>
        </div>
        
        <div class="messages" id="messages">
            <div class="message assistant">
                Olá! Sou o Soul Engine Agent. Ajuste minha personalidade usando os controles à esquerda e vamos conversar! 🧬
            </div>
        </div>
        
        <div class="input-area">
            <div class="input-group">
                <input type="text" id="input" placeholder="Digite sua mensagem..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()" id="sendBtn">Enviar</button>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let sessionId = null;
        
        const presets = {
            neutral: { openness: 50, conscientiousness: 50, extraversion: 50, agreeableness: 50, neuroticism: 50 },
            philosopher: { openness: 90, conscientiousness: 70, extraversion: 30, agreeableness: 60, neuroticism: 40 },
            enthusiast: { openness: 80, conscientiousness: 40, extraversion: 90, agreeableness: 80, neuroticism: 20 },
            analyst: { openness: 60, conscientiousness: 90, extraversion: 30, agreeableness: 40, neuroticism: 30 }
        };
        
        function connect() {
            ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                document.getElementById('status').className = 'status connected';
                console.log('Conectado ao Soul Engine');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'session_id') {
                    sessionId = data.session_id;
                } else if (data.type === 'message') {
                    addMessage('assistant', data.content);
                    document.getElementById('sendBtn').disabled = false;
                }
            };
            
            ws.onclose = () => {
                document.getElementById('status').className = 'status disconnected';
                setTimeout(connect, 3000);
            };
        }
        
        function sendMessage() {
            const input = document.getElementById('input');
            const text = input.value.trim();
            if (!text || !ws) return;
            
            addMessage('user', text);
            
            const personality = {
                openness: parseInt(document.getElementById('openness').value) / 100,
                conscientiousness: parseInt(document.getElementById('conscientiousness').value) / 100,
                extraversion: parseInt(document.getElementById('extraversion').value) / 100,
                agreeableness: parseInt(document.getElementById('agreeableness').value) / 100,
                neuroticism: parseInt(document.getElementById('neuroticism').value) / 100
            };
            
            ws.send(JSON.stringify({ type: 'message', content: text, personality }));
            input.value = '';
            document.getElementById('sendBtn').disabled = true;
        }
        
        function addMessage(role, content) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = content;
            document.getElementById('messages').appendChild(div);
            div.scrollIntoView({ behavior: 'smooth' });
        }
        
        function updateTrait(trait) {
            const value = document.getElementById(trait).value;
            document.getElementById(`val-${trait}`).textContent = value + '%';
        }
        
        function loadPreset(name) {
            const preset = presets[name];
            for (const [trait, value] of Object.entries(preset)) {
                document.getElementById(trait).value = value;
                updateTrait(trait);
            }
        }
        
        function clearChat() {
            document.getElementById('messages').innerHTML = '<div class="message assistant">Chat limpo! Vamos começar de novo. 🔄</div>';
        }
        
        connect();
    </script>
</body>
</html>
"""

@app.get("/")
async def home():
    """Retorna a UI HTML"""
    return HTMLResponse(content=HTML_UI)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para chat em tempo real"""
    await websocket.accept()
    session_id = agent.create_session()

    try:
        # Enviar ID da sessão
        await websocket.send_json({"type": "session_id", "session_id": session_id})

        while True:
            data = await websocket.receive_json()

            if data['type'] == 'message':
                prompt = data['content']
                personality = data.get('personality', {})

                # Atualizar personalidade
                agent.update_personality(session_id, personality)

                # Gerar resposta
                response = await agent.chat(session_id, prompt)

                # Enviar resposta
                await websocket.send_json({"type": "message", "content": response})

    except WebSocketDisconnect:
        print(f"Sessão {session_id} desconectada")

# ============================================================================
# EXECUTAR
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🧬 SOUL ENGINE AGENT")
    print("="*60)
    print("\n🔧 Inicializando sistema...")
    print("\n📝 Quando terminar de carregar:")
    print("   1. Abra seu navegador")
    print("   2. Acesse: http://localhost:8000")
    print("   3. Ajuste os sliders e comece a conversar!")
    print("\n" + "="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
