"""
app.py — Interface Gradio pour la classification SONAR (Mine vs Roche)
Usage : python app.py
"""

import gradio as gr
import numpy as np
import requests
import os

# ============================================================
# URL de l'API FastAPI
# ============================================================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"

EXEMPLE_ROCHE = "0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032"
EXEMPLE_MINE  = "0.0491,0.0279,0.0592,0.1270,0.1772,0.1908,0.2217,0.0768,0.1246,0.2028,0.0947,0.2497,0.2209,0.3195,0.3340,0.3323,0.2780,0.2975,0.2948,0.1729,0.3264,0.3834,0.3523,0.5410,0.5228,0.4475,0.5340,0.5323,0.3907,0.3456,0.4091,0.4639,0.5580,0.5727,0.6355,0.7563,0.6903,0.6176,0.5379,0.5622,0.6508,0.4797,0.3736,0.2804,0.1982,0.2438,0.1789,0.1706,0.0762,0.0238,0.0268,0.0081,0.0129,0.0161,0.0063,0.0119,0.0194,0.0140,0.0332,0.0439"


def parse_signal(text: str):
    import re
    tokens = re.split(r"[,\s;]+", text.strip())
    tokens = [t for t in tokens if t]
    if len(tokens) != 60:
        return None, f"Il faut exactement 60 valeurs — {len(tokens)} détectées."
    try:
        values = [float(t) for t in tokens]
    except ValueError as e:
        return None, f"Valeur non numérique détectée : {e}"
    out_of_range = [v for v in values if not (0.0 <= v <= 1.0)]
    if out_of_range:
        return None, f"{len(out_of_range)} valeur(s) hors de [0, 1]. Vérifiez vos données."
    return np.array(values, dtype=float), None


def make_error_html(message: str) -> str:
    return f"""
    <div class="sonar-error">
        <span style="font-size: 1.4rem;">⚠️</span>
        <span style="font-size: .95rem; font-weight: 500;">{message}</span>
    </div>
    """


def toggle_input_mode(choice):
    """Bascule la visibilité des blocs selon le choix du commutateur."""
    if choice == "✍️ Coller les valeurs":
        return gr.update(visible=True), gr.update(visible=False), 0
    else:
        return gr.update(visible=False), gr.update(visible=True), 1


def predict(signal_text, uploaded_file, active_tab):
    raw_text = None

    if active_tab == 0:
        if signal_text and signal_text.strip():
            raw_text = signal_text.strip()
        else:
            return make_error_html("Veuillez coller des valeurs dans le champ texte.")
    else:
        if uploaded_file is not None:
            try:
                with open(uploaded_file, "r") as f:
                    raw_text = f.read()
            except Exception as e:
                return make_error_html(f"Erreur lecture fichier : {e}")
        else:
            return make_error_html("Veuillez uploader un fichier CSV ou TXT.")

    signal, err = parse_signal(raw_text)
    if err:
        return make_error_html(err)

    if np.all(signal == 0):
        return make_error_html("Signal nul — toutes les valeurs sont à zéro.")

    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json={"frequences": signal.tolist()},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        is_mine    = data["prediction"] == "MINE"
        label      = data["prediction"]
        icon       = "💣" if is_mine else "🪨"
        confidence = data["confiance"] * 100
        p_mine     = data["probabilite_mine"] * 100
        p_roche    = data["probabilite_roche"] * 100

    except requests.exceptions.ConnectionError:
        return make_error_html(f"API inaccessible — vérifie que FastAPI tourne sur {API_URL}")
    except requests.exceptions.Timeout:
        return make_error_html("Timeout — l'API met trop de temps à répondre.")
    except requests.exceptions.HTTPError as e:
        return make_error_html(f"Erreur API ({response.status_code}) : {e}")
    except Exception as e:
        return make_error_html(f"Erreur inattendue : {e}")

    result_html_str = f"""
        <div class="sonar-result {'sonar-mine' if is_mine else 'sonar-roche'}">
            <div class="sonar-corner-deco"></div>

            <div class="sonar-header">
                <div class="sonar-header-main">
                    <div class="sonar-icon-box">{icon}</div>
                    <div>
                        <div class="sonar-label-small">Objet détecté</div>
                        <div class="sonar-label-big">{label}</div>
                    </div>
                </div>
                <div class="sonar-badge">{confidence:.1f}%</div>
            </div>

            <div class="sonar-proba-section">
                <div class="sonar-proba-title">Probabilités</div>

                <div class="sonar-proba-row">
                    <div class="sonar-proba-labels">
                        <span>💣 Mine</span>
                        <span class="sonar-pct-mine">{p_mine:.1f}%</span>
                    </div>
                    <div class="sonar-bar-track">
                        <div class="sonar-bar-fill sonar-bar-mine" style="width:{p_mine:.1f}%"></div>
                    </div>
                </div>

                <div class="sonar-proba-row">
                    <div class="sonar-proba-labels">
                        <span>🪨 Roche</span>
                        <span class="sonar-pct-roche">{p_roche:.1f}%</span>
                    </div>
                    <div class="sonar-bar-track">
                        <div class="sonar-bar-fill sonar-bar-roche" style="width:{p_roche:.1f}%"></div>
                    </div>
                </div>
            </div>
        </div>
        """
    return result_html_str


def load_csv_preview(file):
    if file is None:
        return ""
    try:
        with open(file, "r") as f:
            content = f.read()

        filename = os.path.basename(file)
        total_chars = len(content)
        import re
        tokens = re.split(r"[,\s;]+", content.strip())
        tokens = [t for t in tokens if t]
        n_vals = len(tokens)

        preview = content[:500] + ("…" if total_chars > 500 else "")
        safe = preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        val_ok = n_vals == 60
        val_label = "✓ 60 valeurs" if val_ok else f"⚠ {n_vals} valeur(s)"

        return f"""
        <div class="sonar-preview" style="margin-top: 4px;">
            <div class="sonar-preview-header">
                <div class="sonar-preview-title">
                    <span class="sonar-preview-icon">📄</span>
                    Aperçu
                </div>
                <div class="sonar-preview-badges">
                    <span class="{'sonar-badge-ok' if val_ok else 'sonar-badge-warn'}">{val_label}</span>
                </div>
            </div>
            <div class="sonar-code-block">
                <div class="sonar-code-bar"></div>
                <pre class="sonar-pre"><span class="sonar-code-comment">// {filename}</span>
{safe}</pre>
            </div>
        </div>
        """
    except Exception:
        return make_error_html("Impossible de lire le fichier.")


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

body, .gradio-container {
    font-family: 'Outfit', system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 820px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 12px !important;
}

#input-selector {
    margin-bottom: 16px !important;
    border: none !important;
    background: transparent !important;
}
#input-selector .wrap {
    gap: 8px !important;
}

.gr-button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
}

#analyze-btn {
    height: 54px !important;
    font-size: 1.05rem !important;
    letter-spacing: .02em !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px color-mix(in srgb, var(--primary-500) 25%, transparent);
}

textarea {
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .85rem !important;
}

.sonar-step-container {
    margin: 24px 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid color-mix(in srgb, var(--border-color-primary) 70%, transparent);
    display: flex;
    align-items: center;
    gap: 12px;
}
.sonar-step-num-large {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white !important;
    font-size: 1.1rem;
    font-weight: 800;
    width: 32px; height: 32px;
    border-radius: 9px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 3px 8px rgba(79, 70, 229, 0.3);
    flex-shrink: 0;
}
.sonar-step-text {
    font-size: .95rem;
    font-weight: 700;
    letter-spacing: .02em;
    color: var(--body-text-color);
    text-transform: none;
}

.sonar-main-header { text-align: center; padding: 20px 10px 10px; }
.sonar-main-logo { display: inline-flex; align-items: center; justify-content: center; width: 58px; height: 58px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 16px; font-size: 1.8rem; margin-bottom: 12px; box-shadow: 0 6px 20px rgba(99,102,241,.2); }
.sonar-main-title { font-size: 1.8rem; font-weight: 800; margin: 0 0 6px; background: linear-gradient(135deg, #1e1b4b, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.sonar-main-desc { color: #4b5563; font-size: 0.95rem; margin: 0; }
.sonar-tags-container { display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; justify-content: center; }
.sonar-tag { font-size: .75rem; font-weight: 600; border-radius: 999px; padding: 3px 10px; }

.sonar-error { background: color-mix(in srgb, #dc2626 10%, var(--background-fill-primary)); border: 1.5px solid color-mix(in srgb, #dc2626 40%, transparent); color: var(--body-text-color); border-radius: 12px; padding: 14px 16px; display: flex; align-items: center; gap: 10px; }

.sonar-result { border-radius: 16px; padding: 20px 18px; position: relative; overflow: hidden; }
.sonar-mine { background: color-mix(in srgb, #dc2626 8%, var(--background-fill-primary)); border: 2px solid color-mix(in srgb, #dc2626 35%, transparent); }
.sonar-roche { background: color-mix(in srgb, #16a34a 8%, var(--background-fill-primary)); border: 2px solid color-mix(in srgb, #16a34a 35%, transparent); }
.sonar-corner-deco { position: absolute; top: 0; right: 0; width: 80px; height: 80px; border-radius: 0 16px 0 100%; pointer-events: none; }
.sonar-mine .sonar-corner-deco  { background: color-mix(in srgb, #dc2626 8%, transparent); }
.sonar-roche .sonar-corner-deco { background: color-mix(in srgb, #16a34a 8%, transparent); }

.sonar-header { display: flex; flex-direction: row; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 20px; position: relative; flex-wrap: wrap; }
.sonar-header-main { display: flex; align-items: center; gap: 12px; }
.sonar-icon-box { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0; }
.sonar-mine  .sonar-icon-box { background: color-mix(in srgb, #dc2626 15%, transparent); }
.sonar-roche .sonar-icon-box { background: color-mix(in srgb, #16a34a 15%, transparent); }
.sonar-label-small { font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 1px; color: var(--body-text-color-subdued); }
.sonar-label-big { font-size: 1.6rem; font-weight: 800; line-height: 1.1; }
.sonar-mine  .sonar-label-big { color: #ef4444; }
.sonar-roche .sonar-label-big { color: #22c55e; }
.sonar-badge { color: white; font-weight: 700; font-size: 1rem; border-radius: 8px; padding: 6px 12px; white-space: nowrap; }
.sonar-mine  .sonar-badge { background: #dc2626; }
.sonar-roche .sonar-badge { background: #16a34a; }

.sonar-proba-section { border-top: 1px solid var(--border-color-primary); padding-top: 14px; }
.sonar-proba-title { font-size: .75rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 12px; color: var(--body-text-color-subdued); }
.sonar-proba-row { margin-bottom: 12px; }
.sonar-proba-row:last-child { margin-bottom: 0; }
.sonar-proba-labels { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: .85rem; font-weight: 500; }
.sonar-pct-mine  { font-weight: 700; color: #ef4444; }
.sonar-pct-roche { font-weight: 700; color: #22c55e; }
.sonar-bar-track { background: var(--border-color-primary); border-radius: 999px; height: 8px; overflow: hidden; }
.sonar-bar-fill { height: 100%; border-radius: 999px; }
.sonar-bar-mine  { background: #dc2626; }
.sonar-bar-roche { background: #16a34a; }

.sonar-preview-header { display: flex; flex-direction: row; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
.sonar-preview-title { display: flex; align-items: center; gap: 6px; font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--body-text-color-subdued); }
.sonar-preview-icon { width: 20px; height: 20px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 5px; display: inline-flex; align-items: center; justify-content: center; font-size: .65rem; color: white; }
.sonar-preview-badges { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.sonar-badge-neutral, .sonar-badge-ok, .sonar-badge-warn { font-size: .7rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; display: inline-block; white-space: nowrap; }
.sonar-badge-neutral { background: var(--background-fill-secondary); color: var(--body-text-color); border: 1px solid var(--border-color-primary); max-width: 130px; overflow: hidden; text-overflow: ellipsis; }
.sonar-badge-ok { background: color-mix(in srgb, #16a34a 12%, var(--background-fill-primary)); color: #22c55e; border: 1px solid color-mix(in srgb, #16a34a 35%, transparent); }
.sonar-badge-warn { background: color-mix(in srgb, #dc2626 12%, var(--background-fill-primary)); color: #ef4444; border: 1px solid color-mix(in srgb, #dc2626 35%, transparent); }
.sonar-code-block { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 12px 14px; position: relative; overflow: hidden; }
.sonar-code-bar { position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899); }
.sonar-pre { font-size: .75rem; color: #94a3b8; white-space: pre-wrap; word-break: break-all; margin: 0; line-height: 1.6; font-family: 'JetBrains Mono', monospace; max-height: 120px; overflow-y: auto; }
.sonar-code-comment { color: #818cf8; }

@media (max-width: 576px) {
    .sonar-main-title { font-size: 1.5rem; }
    .sonar-label-big { font-size: 1.4rem; }
    .sonar-header { flex-direction: column; align-items: flex-start; gap: 10px; }
    .sonar-badge { margin-left: 0; align-self: flex-start; }
    .sonar-preview-header { flex-direction: column; align-items: flex-start; }
}
"""

with gr.Blocks(title="SONAR — Mine vs Roche") as demo:

    # État pour suivre la méthode de saisie : 0 = texte, 1 = fichier
    active_tab = gr.State(value=0)

    # ── En-tête Principal ───────────────────────────────────────────────────
    gr.HTML("""
    <div class="sonar-main-header">
        <div class="sonar-main-logo">🔊</div>
        <h1 class="sonar-main-title">SONAR Classifier</h1>
        <p class="sonar-main-desc">
            Analyse de signaux acoustiques pour la détection de menaces sous-marines.
        </p>
        <div class="sonar-tags-container">
            <span class="sonar-tag" style="background:#f0f9ff; color:#0369a1; border:1px solid #bae6fd;">Random Forest</span>
            <span class="sonar-tag" style="background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0;">Dataset UCI · 208 signaux</span>
            <span class="sonar-tag" style="background:#faf5ff; color:#7c3aed; border:1px solid #ddd6fe;">60 features</span>
        </div>
    </div>
    """)

    # ── ÉTAPE 1 ─────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="sonar-step-container">
        <span class="sonar-step-num-large">1</span>
        <span class="sonar-step-text">Méthode d'importation & Données du signal</span>
    </div>
    """)

    input_choice = gr.Radio(
        choices=["✍️ Coller les valeurs", "📂 Uploader un fichier"],
        value="✍️ Coller les valeurs",
        show_label=False,
        elem_id="input-selector",
        type="value"
    )

    # gr.Column(variant="panel") remplace gr.Box obsolète tout en gardant une jolie boîte
    with gr.Column(variant="panel", visible=True) as box_text:
        signal_text = gr.Textbox(
            label="",
            show_label=False,
            placeholder="Entrez les 60 coefficients séparés par des virgules (ex: 0.0200, 0.0371...)",
            lines=4,
            max_lines=8,
        )
        with gr.Row():
            btn_mine  = gr.Button("💣  Exemple Mine",  variant="secondary", min_width=120)
            btn_roche = gr.Button("🪨  Exemple Roche", variant="secondary", min_width=120)
            btn_clear = gr.Button("🗑️  Effacer",       variant="stop",      min_width=80)

    with gr.Column(variant="panel", visible=False) as box_file:
        uploaded_file = gr.File(
            label="Fichier CSV ou TXT (60 valeurs délimitées)",
            file_types=[".csv", ".txt"],
            type="filepath",
        )
        file_preview = gr.HTML(value="")
        uploaded_file.change(fn=load_csv_preview, inputs=uploaded_file, outputs=file_preview, queue=False)

    input_choice.change(
        fn=toggle_input_mode,
        inputs=[input_choice],
        outputs=[box_text, box_file, active_tab],
        queue=False
    )

    # ── ÉTAPE 2 ─────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="sonar-step-container">
        <span class="sonar-step-num-large">2</span>
        <span class="sonar-step-text">Exécution des algorithmes de prédiction</span>
    </div>
    """)

    btn_predict = gr.Button(
        "🔍  Lancer l'analyse du signal",
        variant="primary",
        size="lg",
        elem_id="analyze-btn",
    )

    with gr.Row(visible=True, elem_id="result-area") as result_row:
        result_html = gr.HTML(value="")

    # ── Liaisons des actions ─────────────────────────────────────────────────
    btn_predict.click(
        fn=predict,
        inputs=[signal_text, uploaded_file, active_tab],
        outputs=[result_html],
        queue=False,
    )

    btn_mine.click(fn=lambda: EXEMPLE_MINE,   outputs=signal_text, queue=False)
    btn_roche.click(fn=lambda: EXEMPLE_ROCHE, outputs=signal_text, queue=False)
    btn_clear.click(fn=lambda: "",             outputs=signal_text, queue=False)


if __name__ == "__main__":
    # Correction : Passage de css=CSS dans launch() comme requis à partir de Gradio 4/5/6
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=CSS)