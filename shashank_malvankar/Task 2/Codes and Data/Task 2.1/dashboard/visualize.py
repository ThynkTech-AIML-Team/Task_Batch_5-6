import pyLDAvis
import pyLDAvis.lda_model
import os

def save_lda_visualization(lda_model, X, vectorizer,
                           lda_perplexity,
                           lda_coherence,
                           nmf_coherence):

    print("Preparing LDA visualization...")

    vis = pyLDAvis.lda_model.prepare(
        lda_model,
        X,
        vectorizer
    )

    html_vis = pyLDAvis.prepared_data_to_html(vis)

    scores_html = f"""
    <h2>Model Comparison Scores</h2>
    <ul>
        <li><b>LDA Perplexity:</b> {lda_perplexity:.4f}</li>
        <li><b>LDA Coherence:</b> {lda_coherence:.4f}</li>
        <li><b>NMF Coherence:</b> {nmf_coherence:.4f}</li>
    </ul>
    <hr>
    """

    full_html = scores_html + html_vis

    output_path = "lda_dashboard.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print("Dashboard saved at:", os.path.abspath(output_path))
