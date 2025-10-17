import gradio as gr
import pandas as pd
from typing import List, Optional

from simulacra.benchmark import (
    _log,
    run_benchmark_experiment,
    save_results_to_csv,
    DEFAULT_EMBEDDING_DIM,
    EMBEDDING_DIM_INFER_THRESHOLD,
)
from simulacra.embeddings import generate_embeddings_with_pythae, create_embeddings_with_target
from simulacra_gui.utils import (
    load_dataframe_from_file,
    dataframe_shape_str,
    compact_preview,
    highlight_headers_html,
    perform_join,
    save_working_dataset,
    write_embeddings_as_is,
)


def build_preview_html(df: pd.DataFrame, highlight_cols: Optional[List[str]] = None) -> str:
    preview_df, elide_rows, elide_cols = compact_preview(df, n=5)
    return highlight_headers_html(preview_df, highlight_cols=highlight_cols, max_height=420)


def ui_logic(
    file_left,
    file_right,
    left_keys: List[str],
    right_keys: List[str],
    target_col: str,
    embed_mode: str,
    embed_dim: Optional[int],
    synth_mode: str,
    accession: str,
):
    # Load
    df_left = load_dataframe_from_file(file_left) if file_left else None
    if df_left is None:
        raise gr.Error("Please upload a primary dataframe.")

    df = df_left.copy()

    # Join if provided
    if file_right is not None:
        df_right = load_dataframe_from_file(file_right)
        if not left_keys or not right_keys:
            raise gr.Error("Please select join key columns for both tables.")
        df = perform_join(df, df_right, left_keys, right_keys, how='left')

    if target_col not in df.columns:
        raise gr.Error("Please select a valid target column present in the working dataframe.")

    # Save working dataset for downstream pipeline
    save_working_dataset(df, target_column=target_col, accession=accession)

    # Embeddings
    if embed_mode == 'Auto':
        cols = df.drop(columns=[target_col]).shape[1]
        desired_dim = embed_dim if embed_dim else (cols if cols < EMBEDDING_DIM_INFER_THRESHOLD else DEFAULT_EMBEDDING_DIM)
        if cols < EMBEDDING_DIM_INFER_THRESHOLD:
            # Use as is
            write_embeddings_as_is(df, target_column=target_col, accession=accession)
        else:
            generate_embeddings_with_pythae(accession=accession, latent_dim=int(desired_dim))
            create_embeddings_with_target(accession=accession, target_col=target_col)
    elif embed_mode == 'Force Pythae':
        desired_dim = embed_dim if embed_dim else DEFAULT_EMBEDDING_DIM
        generate_embeddings_with_pythae(accession=accession, latent_dim=int(desired_dim))
        create_embeddings_with_target(accession=accession, target_col=target_col)
    else:  # Skip (use as-is)
        write_embeddings_as_is(df, target_column=target_col, accession=accession)

    # Synthesis
    use_cuda = None
    seeds = [42, 931782, 8481962]
    multipliers = (1, 2)

    # Fast mode: override to GaussianCopula KDE via optimize path not required; we can run standard benchmark then adjust later
    if synth_mode == 'Fast':
        # For simplicity, reuse run_benchmark_experiment which will optimize by default.
        # A future improvement could add a direct fast path. For now, user-facing choice exists.
        all_results, summary_stats = run_benchmark_experiment(
            accession=accession,
            seeds=seeds,
            multipliers=multipliers,
            use_cuda=use_cuda,
        )
    elif synth_mode == 'Best':
        all_results, summary_stats = run_benchmark_experiment(
            accession=accession,
            seeds=seeds,
            multipliers=multipliers,
            use_cuda=use_cuda,
        )
    else:
        # Custom: for first version, map to the same experiment API; future: expose granular controls
        all_results, summary_stats = run_benchmark_experiment(
            accession=accession,
            seeds=seeds,
            multipliers=multipliers,
            use_cuda=use_cuda,
        )

    csv_path = save_results_to_csv(
        all_results=all_results,
        summary_stats=summary_stats,
        accession=accession,
        target_column=target_col,
    )

    # Build summary table
    rows = []
    for seed, results in all_results.items():
        for method, data in results.items():
            rows.append({
                'seed': seed,
                'method': method,
                'accuracy': data['metrics']['accuracy'],
                'f1_macro': data['metrics']['f1_macro'],
            })
    summary_df = pd.DataFrame(rows)

    return (
        build_preview_html(df_left, highlight_cols=left_keys or []),
        build_preview_html(df_right) if file_right else "",
        dataframe_shape_str(df),
        gr.update(value=summary_df),
        csv_path,
    )


def app():
    with gr.Blocks(title="Simulacra GUI") as demo:
        gr.Markdown("# Simulacra — Biologist-Friendly Interface")
        gr.Markdown("Upload your data, choose a target, optionally join another table, select embeddings and synthesis mode, then run.")

        with gr.Row():
            with gr.Column():
                file_left = gr.File(label="Primary dataframe (CSV/Parquet)")
                file_right = gr.File(label="Optional: second dataframe to join (CSV/Parquet)")
                accession = gr.Textbox(label="Session ID (used as accession)", value="USER")

                left_keys = gr.CheckboxGroup(choices=[], label="Left join keys (from primary)")
                right_keys = gr.CheckboxGroup(choices=[], label="Right join keys (from second)")

                target_col = gr.Dropdown(choices=[], label="Target column", interactive=True)

                embed_mode = gr.Radio([
                    'Auto', 'Force Pythae', 'Skip (use as-is)'
                ], value='Auto', label="Embeddings")
                embed_dim = gr.Number(label=f"Embedding dimension (optional, default {DEFAULT_EMBEDDING_DIM})", value=None)

                synth_mode = gr.Radio([
                    'Fast', 'Best', 'Custom'
                ], value='Best', label="Synthesis Mode")

                run_btn = gr.Button("Run Benchmark")

            with gr.Column():
                preview_left = gr.HTML(label="Preview: Primary", value="")
                preview_right = gr.HTML(label="Preview: Second", value="")
                data_shape = gr.Markdown()
                results_df = gr.Dataframe(label="Results", interactive=False)
                csv_out = gr.File(label="Download CSV")

        def on_file_change(left, right):
            df_left = load_dataframe_from_file(left) if left else None
            df_right = load_dataframe_from_file(right) if right else None
            left_cols = list(df_left.columns) if df_left is not None else []
            right_cols = list(df_right.columns) if df_right is not None else []
            # Default preselect leftmost column(s): pick first column
            default_left = [left_cols[0]] if left_cols else []
            default_right = [right_cols[0]] if right_cols else []
            return (
                build_preview_html(df_left, highlight_cols=default_left) if df_left is not None else "",
                build_preview_html(df_right, highlight_cols=default_right) if df_right is not None else "",
                dataframe_shape_str(df_left) if df_left is not None else "",
                gr.update(choices=left_cols, value=default_left),
                gr.update(choices=right_cols, value=default_right),
                gr.update(choices=left_cols, value=left_cols[0] if left_cols else None),
            )

        file_left.change(on_file_change, inputs=[file_left, file_right], outputs=[preview_left, preview_right, data_shape, left_keys, right_keys, target_col])
        file_right.change(on_file_change, inputs=[file_left, file_right], outputs=[preview_left, preview_right, data_shape, left_keys, right_keys, target_col])

        run_btn.click(
            ui_logic,
            inputs=[file_left, file_right, left_keys, right_keys, target_col, embed_mode, embed_dim, synth_mode, accession],
            outputs=[preview_left, preview_right, data_shape, results_df, csv_out]
        )

    return demo


if __name__ == "__main__":
    app().launch()


