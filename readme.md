

# 🧬 Soul Engine: Geometric Personality Steering

**Soul Engine** is a modular framework for **Representation Engineering (RepEng)**. It allows for the deterministic manipulation of Large Language Model (LLM) behavior by injecting specific direction vectors into the model's residual stream during inference.

Unlike Fine-Tuning (SFT/LoRA), which permanently alters model weights, Soul Engine performs **non-destructive, inference-time surgery**. By identifying the linear subspace where concepts like "Refusal," "Sarcasm," or "Introversion" reside, we can mathematically add or subtract these traits from the model's "mind" in real-time.

<img width="1330" height="847" alt="image" src="https://github.com/user-attachments/assets/d937a97c-33f3-4d43-a78b-7ef34b2a3bf4" />


---

## 🚀 Key Features

*   **Zero-Shot Personality Injection:** Instantly turn a base model into a pirate, a philosopher, or an anime character without training.
*   **Safety Lobotomy (Jailbreak):** Mathematically subtract the "Refusal" vector (Safety/RLHF) to bypass guardrails, or amplify it to create a Puritan model.
*   **Soul Scanner:** An automated diagnostic tool that sweeps through transformer layers to find the "Sweet Spot" (optimal layer) for vector injection.
*   **MBTI Forge:** A programmatic engine to construct complex personalities based on the 4-axis Myers-Briggs Type Indicator.
*   **Soul Cards (.soul):** A portable file format to save, share, and load extracted personality vectors.
*   **Gradio Studio:** A full-featured Web UI for forging, testing, and managing personalities.

---

## 📂 Architecture Overview

The project is structured as a class hierarchy, culminating in the Web UI.

### 1. `soul_engine.py` (The Kernel)
The base class.
*   **Function:** Loads the LLM (default: `Qwen/Qwen2.5-0.5B-Instruct`).
*   **Core Logic:** Implements `extract_personality_vector` (Difference of Means) and `generate_steered` (PyTorch Forward Hook injection).
*   **Math:** $h' = h + \alpha \cdot \vec{v}_{concept}$

### 2. `soul_forge.py` (The Alchemist)
Inherits from `SoulEngine`.
*   **Function:** Handles complex vector arithmetic.
*   **MBTI Logic:** Contains the definitions for E/I, N/S, T/F, J/P axes. It constructs a persona by summing these orthogonal vectors:
    *   $\vec{v}_{INTJ} = (\vec{v}_{Introversion} + \vec{v}_{Intuition} + \vec{v}_{Thinking} + \vec{v}_{Judging})$

### 3. `soul_cards.py` (The Librarian)
Inherits from `SoulForge`.
*   **Function:** Manages persistence (I/O).
*   **Soul Format:** Saves vectors + metadata (Layer ID, Name, Description) into `.soul` files (serialized Torch tensors).
*   **Deck Management:** Lists, loads, and uploads cards.

### 4. `app.py` (The Interface)
The Gradio frontend.
*   **Function:** Provides a dashboard for the entire system.
*   **Features:** Chatbot with real-time sliders (Strength/Layer), "Grimoire" for file management, and "Forge" for creating new cards.

---

## 🛠️ Installation & Usage

### Prerequisites
You need Python 3.10+ and a GPU (recommended), though it falls back to CPU.

```bash
pip install torch transformers accelerate gradio numpy scikit-learn
```

### Running the Studio
Launch the unified interface:

```bash
python app.py
```

Open your browser at the provided local URL (usually `http://127.0.0.1:7860`).

---

## 🕹️ How to Use the Studio

### 1. The Chat (Personality Test)
*   **Equip a Card:** Go to the "Grimoire" tab and load a card.
*   **Strength Slider:** Controls the intensity ($\alpha$).
    *   `+4.0`: Standard injection.
    *   `+10.0`: Extreme caricature (may cause hallucinations).
    *   `-4.0`: Injects the *opposite* trait (Reverse Steering).
*   **Layer Override:**
    *   `0`: Uses the layer saved inside the card.
    *   `1-24`: Forces the vector into a specific layer for experimentation.

### 2. The Grimoire (Management)
*   **Load:** Equip existing `.soul` cards.
*   **Diagnostic Scanner:** Click **"Microscope"** to test the equipped card against all layers. If the scanner finds a layer with better distinctiveness than the original, it will automatically adjust your Chat slider.
*   **Import/Export:** Upload `.soul` files from friends or download your current creation.

### 3. Custom Forge
Create a specific trait (e.g., "Dark Wizard").
1.  **Positive Samples:** Input 5-10 phrases representing the concept (e.g., "I cast a fireball", "Magic is power").
2.  **Negative Samples:** Input 5-10 phrases of the opposite (e.g., "I use a sword", "Science is truth").
3.  **Scan:** Click **"Scan Layers"**. The system will identify which layer creates the strongest separation between these concepts.
4.  **Craft:** Saves the result as a `.soul` file.

---

## ⚙️ Advanced: Editing MBTI Definitions

The MBTI system relies on "Anchor Sentences" to define what "Introversion" or "Thinking" means to the model. You can manually refine these definitions to improve the quality of generated personas.

**File:** `soul_forge.py`

Locate the `build_mbti_vectors` method. You will see the `axes_data` dictionary:

```python
# soul_forge.py

axes_data = {
    "E_vs_I": (
        # POSITIVE (Extraversion)
        ["I love parties!", "I speak before I think.", "Action is better than reflection."], 
        # NEGATIVE (Introversion)
        ["I enjoy solitude.", "I think before I speak.", "Reflection is better than action."]        
    ),
    # ... other axes ...
}
```

### How to customize:
1.  **Add more examples:** The more sentences, the more accurate the vector direction.
2.  **Change the nuance:**
    *   *Current:* Generic Introversion.
    *   *Mod:* Anxious Introversion. Change the negative samples to: *"I am afraid of people," "Socializing makes me panic."*
    *   *Result:* Any "I" type generated (INTJ, INFP) will now inherit this anxiety.
3.  **Delete `cards/*.soul`:** After changing the code, delete existing MBTI cards so the system recalculates the vectors next time you run `Craft MBTI`.

---

## 🔬 Technical Theory

	arXiv:2512.07092 
   The Geometry of Persona: Disentangling Personality from Reasoning in Large Language Models

   
This project implements the **Linear Representation Hypothesis**:
> "Deep neural networks represent concepts as linear directions in their activation space."

By extracting the mean difference between two sets of activations:
$$\vec{v}_{concept} = \frac{1}{N}\sum E(x_{pos}) - \frac{1}{N}\sum E(x_{neg})$$

We obtain a steering vector. Injecting this vector shifts the model's latent state towards the concept.

*   **Layer Anatomy (Qwen 0.5B):**
    *   **Layers 0-5:** Syntax & Surface form (Injecting here affects grammar).
    *   **Layers 10-16 (The Sweet Spot):** Semantics, Intent, Safety, & Personality.
    *   **Layers 20+:** Output formatting (Injecting here often has little effect).

---

## ⚠️ Disclaimer

This tool allows for the removal of safety guardrails (RLHF) via geometric ablation. It is intended for **research purposes only** in the fields of Mechanistic Interpretability and AI Alignment. The authors are not responsible for output generated by modified models.
