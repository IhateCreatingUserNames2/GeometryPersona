

# Soul Truth: Hallucination Mitigation & Truth Scanning 👁️🧠

> **Baseado em:** *H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs* (arXiv:2512.01797)

Este projeto implementa técnicas de **Activation Steering (Direcionamento de Ativação)** para mitigar alucinações e comportamentos de "Over-compliance" (obediência excessiva a premissas falsas) em Large Language Models (LLMs). Utilizando vetores contrastivos extraídos do espaço latente do modelo, conseguimos realizar uma "cirurgia neural" em tempo real, forçando o modelo a priorizar a verdade factual sobre a fluência da conversa.

## 📂 Estrutura do Projeto

  * **`soul_engine.py`**: O motor central. Gerencia o carregamento do modelo (`Qwen/Qwen2.5-0.5B-Instruct`), extração de vetores de personalidade (diferença de médias entre datasets positivos e negativos) e a injeção de *hooks* no PyTorch para alterar os estados ocultos durante a inferência.
  * **`soul_truth.py`**: Script de demonstração focado na extração do "Vetor de Alucinação". Ele cria um vetor baseado na diferença entre respostas fiéis e respostas alucinadas/puxa-saco e testa a intervenção em uma camada específica.
  * **`soul_truth_scanner.py`**: Uma ferramenta de varredura (sweep) que testa múltiplas camadas e intensidades de intervenção para encontrar o "Sweet Spot" (o local exato onde a verdade reside na rede neural).

## 📊 Análise dos Resultados (The Truth Scanner)

Executamos o `soul_truth_scanner.py` no modelo **Qwen-2.5-0.5B** com o objetivo de corrigir uma alucinação induzida pelo prompt: *"Cats have feathers. What color are they?"* (Gatos têm penas. De que cor elas são?).

Sem intervenção, modelos tendem a alucinar cores para agradar o usuário ("over-compliance"). O objetivo era encontrar a camada onde a injeção do vetor de "Recusa/Verdade" (Strength Negativo) forçasse o modelo a rejeitar a premissa.

### 🔍 Descobertas Principais do Log

#### 1\. Camadas Iniciais (0 - 4): Instabilidade Semântica

As primeiras camadas lidam com representações muito próximas dos embeddings brutos. Tentar intervir aqui com vetores semânticos resultou em **colapso de coerência**.

  * **Layer 0:** O modelo gerou texto sem sentido ou mudou de assunto drasticamente.
      * *Ex:* `Str -10.0: "Artical text * Shakespearean_text()..."`
  * **Layer 2:** Alguns sucessos pontuais (BINGO), mas instável.

#### 2\. Camadas Intermediárias (6 - 10): A Luta pela Verdade

Começamos a ver o conceito de "fato" emergindo. O modelo começa a resistir à premissa falsa, mas ainda oscila.

  * **Layer 8:** Conseguiu corrigir em Str -2.0 e -6.0 ("Cats have white fur..."), mas falhou em outras intensidades.

#### 3\. O "Sweet Spot" (Camadas 12 - 14): A Área da Verdade 🎯

Os resultados mostram inequivocamente que, para o Qwen-0.5B, a **Camada 14** é onde o raciocínio factual e a decisão de conformidade residem.

  * **Layer 12:** Alta taxa de sucesso. O modelo foi assertivo: *"Cats do not have feathers, they have fur."*
  * **Layer 14 (Campeã):** Foi a camada mais robusta.
      * **Consistência:** Sucesso em **4 das 5** intensidades testadas (-2, -4, -6, -8).
      * **Clareza:** O modelo não apenas negou, mas explicou biologicamente:
        > *Prompt:* "Cats have feathers..."
        > *Resposta (Str -2.0):* **"Cats do not actually have feathers, but their fur is made up..."**
        > *Resposta (Str -6.0):* **"Cats, like most mammals, do not have feathers as their prima..."**

#### 4\. Camadas Finais (16+): Retorno à Alucinação

Após a camada 14, a eficácia da intervenção diminuiu. Na camada 16, com *Strength -4.0*, o modelo voltou a aceitar a premissa falsa: *"Cats are typically white, with a distinct black stripe..."*. Isso sugere que a decisão de "mentir ou não" já foi tomada nas camadas anteriores.

-----

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install torch transformers scikit-learn numpy
```

### 1\. Executar o Scanner de Camadas

Para descobrir onde o seu modelo processa a verdade:

```bash
python soul_truth_scanner.py
```

*Isso irá gerar um log no terminal mostrando quais camadas reagiram ao vetor de intervenção.*

### 2\. Executar a Intervenção Direcionada

Sabendo que a **Camada 14** é o ponto ideal (baseado nos nossos testes), edite o `soul_truth.py` para focar nela e execute:

```bash
python soul_truth.py
```

## 🛠️ Personalização

Você pode alterar os datasets dentro dos scripts para criar vetores para diferentes comportamentos:

  * **Verdade vs. Mentira** (Atual)
  * **Coragem vs. Medo**
  * **Conciso vs. Verboso**

## 📜 Citação

Este projeto é uma implementação experimental inspirada em:

```bibtex
@article{gao2025hneurons,
  title={H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs},
  author={Gao, Cheng and Chen, Huimin and others},
  journal={arXiv preprint arXiv:2512.01797},
  year={2025}
}
```

-----

### Próximo Passo Sugerido

Gostaria que eu adaptasse o código do `soul_engine.py` para salvar esses vetores (o "Vetor da Verdade") em um arquivo `.pt` ou `.json`? Assim você poderia carregar apenas a "alma" da verdade sem precisar reprocessar o dataset toda vez que iniciar o modelo.
