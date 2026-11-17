"""Validated run report and fold-specific explanations, separate from legacy demo."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import config
from src.artifact_registry import load_validated_run
from src.explainability import explain_fold, driver_table
from src.scorecard import pd_to_score
from src.reporting import borrower_assessment, model_card


@st.cache_resource
def load_run(path, revision):
    return load_validated_run(path)


@st.cache_data
def explanations(path, revision, fold):
    bundle, sample, _, _, _ = load_run(path, revision)
    return explain_fold(bundle, sample.drop(columns='SK_ID_CURR'), fold)


def render():
    st.title('Model validation and explanations')
    st.caption('Research demo • Custom model scores • Independent evaluation within the Home Credit dataset')
    root = Path(config.ARTIFACTS_DIR) / 'validated'
    runs = sorted(p for p in root.glob('*') if p.is_dir() and (p / 'manifest.json').exists())
    if not runs:
        st.info('No completed validated run is available. Run python -m src.validated_training to create one.')
        return
    chosen = st.selectbox('Validated run', runs, format_func=lambda p: p.name)
    # Immutable run folders are expected; file stats also invalidate caches on replacement.
    required = ['manifest.json', 'inference_bundle.pkl', 'example_features.pkl', 'test_predictions.csv', 'metrics.json']
    try:
        revision = tuple((f, (chosen / f).stat().st_mtime_ns, (chosen / f).stat().st_size) for f in required)
        with st.spinner('Checking model and saved predictions...'):
            bundle, sample, predictions, report, manifest = load_run(str(chosen), revision)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'This run could not be verified: {exc}')
        return
    st.caption(f"Run: {chosen.name} · Trained: {manifest['created_utc']} · Source: {manifest['git_commit'][:10]}" +
               (' with local changes' if manifest.get('working_tree_dirty') else ''))
    validation, local, global_view, card_view = st.tabs(['Validation results', 'Why this risk?', 'Sample feature importance', 'Model card'])
    with card_view:
        card = model_card(chosen.name, report, manifest)
        st.download_button('Download model card', card, f'{chosen.name}_model_card.md', 'text/markdown')
        st.markdown(card)
    with validation:
        a, b, c = st.columns(3)
        for col, name in zip((a, b, c), ('development', 'calibration', 'test')):
            col.metric(name.title() + ' borrowers', f"{report['cohort_sizes'][name]:,}")
        st.write(f"Test default rate: {report['default_rate']:.2%} ({report['defaults']:,} defaults).")
        st.dataframe(pd.DataFrame({'Raw ensemble': report['raw'], 'Calibrated ensemble': report['calibrated']}), width='stretch')
        st.caption('Higher ROC-AUC and average precision are better. Lower Brier score and log loss are better. Average precision is not trapezoidal PR area.')
        if all(report['calibrated'][m] >= report['raw'][m] for m in ('brier_score', 'log_loss')):
            st.warning('Calibration did not improve Brier score or log loss on this test cohort. Both results are retained; no model was selected using these test outcomes.')
        fig = go.Figure()
        for kind, curve in report['reliability'].items():
            fig.add_scatter(x=curve['mean_prediction'], y=curve['observed_rate'], mode='lines+markers', name=kind.title())
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect calibration', line={'dash': 'dash'})
        fig.update_layout(title='Held-out reliability', xaxis_title='Mean predicted probability', yaxis_title='Observed default rate')
        st.plotly_chart(fig, width='stretch')
        with st.expander('95% bootstrap intervals and limitations'):
            rows = [{'Model': kind, 'Metric': metric, 'Lower': bounds[0], 'Upper': bounds[1]}
                    for kind, intervals in report['confidence_intervals_95'].items() for metric, bounds in intervals.items()]
            st.dataframe(pd.DataFrame(rows), width='stretch')
            st.write(report['bootstrap']['scope'])
            for limitation in report['limitations']:
                st.write('• ' + limitation)
        st.download_button('Download validation report', json.dumps(report, indent=2), 'validation_report.json', 'application/json')
    # Compute only the selected fold, once per run/fold; no full-portfolio SHAP work.
    with local:
        st.write(f'Choose from {len(sample)} saved held-out examples with verified model inputs.')
        borrower = st.selectbox('Borrower ID', sample.SK_ID_CURR.tolist())
        fold = st.selectbox('Fold model to explain', list(range(len(bundle['fold_models']))), format_func=lambda i: f'Fold {i + 1}')
        try:
            with st.spinner('Computing fold-specific Tree SHAP explanations...'):
                result = explanations(str(chosen), revision, fold)
        except (ValueError, TypeError) as exc:
            st.error(f'Explanation integrity check failed: {exc}')
            return
        index = sample.index[sample.SK_ID_CURR == borrower][0]
        prediction = predictions.loc[predictions.SK_ID_CURR == borrower].iloc[0]
        a, b, c = st.columns(3)
        a.metric('Full ensemble raw PD', f'{prediction.PD_RAW:.2%}')
        b.metric('Full ensemble calibrated PD', f'{prediction.PD_CALIBRATED:.2%}')
        c.metric('Custom model score (300–850)', f'{float(pd_to_score(prediction.PD_CALIBRATED)):.0f}')
        st.info(f"The charts explain fold {fold + 1} only, in raw log-odds. Its probability is {result['fold_probability'][index]:.2%}. Contributions do not add up to the ensemble's raw or calibrated PD. They describe model associations, not causes or lending decisions.")
        table = driver_table(sample.loc[index, bundle['feature_names']], result['values'][index])
        top = table.head(10)
        other = float(table.iloc[10:]['SHAP contribution'].sum())
        fig = go.Figure(go.Waterfall(x=['Expected margin'] + top.Feature.tolist() + ['Other features', 'Fold margin'],
                       y=[float(result['base_values'][index])] + top['SHAP contribution'].tolist() + [other, 0],
                       measure=['absolute'] + ['relative'] * (len(top) + 1) + ['total']))
        fig.update_layout(title=f'Fold {fold + 1} contribution waterfall', yaxis_title='Raw log-odds', height=520)
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Reconstruction checked: base + all contributions = {result['raw_margin'][index]:.6f} raw log-odds. Positive contributions raise this fold's estimated risk; negative contributions lower it.")
        st.subheader('Five largest drivers')
        st.dataframe(table.head(5), width='stretch', hide_index=True)
        st.download_button('Download all borrower drivers', table.to_csv(index=False), f'borrower_{borrower}_fold_{fold + 1}_drivers.csv', 'text/csv')
        assessment = borrower_assessment(chosen.name, manifest, prediction, sample.loc[index], table, result, index)
        st.download_button('Download borrower assessment', assessment, f'borrower_{borrower}_assessment.md', 'text/markdown')
        with st.expander('Preview borrower assessment and loss assumptions'):
            st.markdown(assessment)
    with global_view:
        st.caption(f'Mean absolute Tree SHAP values for fold {fold + 1}, over {len(sample)} saved test examples. This small sample is not a portfolio-wide importance estimate.')
        importance = pd.DataFrame({'Feature': bundle['feature_names'], 'Mean absolute SHAP': np.abs(result['values']).mean(axis=0)})
        importance = importance.nlargest(20, 'Mean absolute SHAP').sort_values('Mean absolute SHAP')
        fig = px.bar(importance, x='Mean absolute SHAP', y='Feature', orientation='h', title='Top features in the saved sample')
        fig.update_layout(height=650)
        st.plotly_chart(fig, width='stretch')
