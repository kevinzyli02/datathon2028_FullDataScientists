import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def generate_pls_report(results, data, output_dir="pls_report"):
    """Generate a comprehensive PLS analysis report"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create report content
    report_content = generate_report_content(results, data)

    # Save text report
    with open(f"{output_dir}/pls_analysis_report.txt", "w") as f:
        f.write(report_content)

    # Generate visualizations
    generate_visualizations(results, data, output_dir)

    # Create summary CSV files
    generate_summary_files(results, output_dir)

    print(f"✅ Report generated in '{output_dir}' directory")
    return report_content


def generate_report_content(results, data):
    """Generate the main report content"""

    report = []
    report.append("=" * 80)
    report.append("PLS ANALYSIS REPORT - HORMONE PREDICTION")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 1. Executive Summary
    report.append("1. EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Number of observations: {results['X'].shape[0]}")
    report.append(f"Number of predictors used: {len(results['predictors'])}")
    report.append(f"Number of target hormones: {len(results['targets'])}")
    report.append("")

    # 2. Model Performance
    report.append("2. MODEL PERFORMANCE")
    report.append("-" * 40)
    report.append("R² Scores (Variance Explained):")
    for hormone, r2 in results['r2_scores'].items():
        interpretation = interpret_r2(r2)
        report.append(f"  {hormone.upper():<10} : {r2:.4f} - {interpretation}")
    report.append("")

    # 3. Feature Importance
    report.append("3. FEATURE IMPORTANCE (VIP SCORES)")
    report.append("-" * 40)
    report.append("VIP Score Interpretation:")
    report.append("  VIP > 1.0: Highly important")
    report.append("  VIP 0.8-1.0: Moderately important")
    report.append("  VIP < 0.8: Less important")
    report.append("")

    importance_df = results['importance_df']
    report.append("Top 15 Most Important Features:")
    for i, (idx, row) in enumerate(importance_df.head(15).iterrows()):
        importance_level = "***" if row['VIP_score'] > 1.0 else "** " if row['VIP_score'] > 0.8 else "*  "
        report.append(f"  {i + 1:2d}. {importance_level} {row['feature']:<20} : {row['VIP_score']:.4f}")
    report.append("")

    # 4. Data Quality Summary
    report.append("4. DATA QUALITY SUMMARY")
    report.append("-" * 40)
    report.append("Predictor Variables:")
    for predictor in results['predictors']:
        non_missing = data[predictor].notna().sum()
        total = len(data)
        pct_missing = (1 - non_missing / total) * 100
        report.append(f"  {predictor:<20} : {non_missing}/{total} ({pct_missing:.1f}% missing)")
    report.append("")

    report.append("Target Variables:")
    for target in results['targets']:
        non_missing = data[target].notna().sum()
        total = len(data)
        pct_missing = (1 - non_missing / total) * 100
        report.append(f"  {target:<10} : {non_missing}/{total} ({pct_missing:.1f}% missing)")
    report.append("")

    # 5. Biological Insights
    report.append("5. BIOLOGICAL INSIGHTS")
    report.append("-" * 40)
    report.append(generate_biological_insights(results))
    report.append("")

    # 6. Recommendations
    report.append("6. RECOMMENDATIONS")
    report.append("-" * 40)
    report.append(generate_recommendations(results))
    report.append("")

    # 7. Technical Details
    report.append("7. TECHNICAL DETAILS")
    report.append("-" * 40)
    report.append(f"PLS Components used: {results['pls_model'].n_components}")
    report.append(f"Total features considered: {len(results['predictors'])}")
    report.append(f"Final features used: {len(results['X'].columns)}")
    report.append(f"Analysis method: Partial Least Squares Regression")
    report.append("")

    return "\n".join(report)


def interpret_r2(r2):
    """Interpret R² values"""
    if r2 >= 0.7:
        return "Excellent prediction"
    elif r2 >= 0.5:
        return "Good prediction"
    elif r2 >= 0.3:
        return "Moderate prediction"
    elif r2 >= 0.1:
        return "Weak prediction"
    else:
        return "Very weak prediction"


def generate_biological_insights(results):
    """Generate biological insights based on feature importance"""
    insights = []
    top_features = results['importance_df'].head(10)['feature'].tolist()

    # Categorize features
    sleep_features = [f for f in top_features if any(x in f for x in ['sleep', 'minutesasleep', 'efficiency'])]
    symptom_features = [f for f in top_features if
                        f in ['appetite', 'exerciselevel', 'headaches', 'cramps', 'fatigue', 'stress', 'bloating']]
    metabolic_features = [f for f in top_features if 'glucose' in f]

    insights.append("Key Feature Categories in Top Predictors:")
    if sleep_features:
        insights.append(f"  • Sleep-related: {', '.join(sleep_features)}")
    if symptom_features:
        insights.append(f"  • Self-reported symptoms: {', '.join(symptom_features)}")
    if metabolic_features:
        insights.append(f"  • Metabolic: {', '.join(metabolic_features)}")

    # Specific insights
    if 'stress' in top_features[:3]:
        insights.append("  → Stress appears to be a key hormonal regulator")
    if 'glucose_mean' in top_features[:5]:
        insights.append("  → Glucose levels show significant relationship with hormones")
    if any(f in top_features[:3] for f in ['fatigue', 'sleepissue']):
        insights.append("  → Sleep quality and fatigue are important predictors")

    return "\n".join(insights)


def generate_recommendations(results):
    """Generate recommendations based on analysis results"""
    recommendations = []

    r2_values = list(results['r2_scores'].values())
    avg_r2 = np.mean(r2_values)

    if avg_r2 < 0.1:
        recommendations.append("• Consider collecting additional predictor variables")
        recommendations.append("• Investigate data quality issues in current predictors")
        recommendations.append("• Explore non-linear relationships or interaction terms")
    elif avg_r2 < 0.3:
        recommendations.append("• Model shows potential but needs improvement")
        recommendations.append("• Focus on the top predictors identified")
        recommendations.append("• Consider time-lagged effects of predictors")
    else:
        recommendations.append("• Model shows good predictive capability")
        recommendations.append("• Top predictors should be prioritized in future studies")
        recommendations.append("• Consider clinical validation of findings")

    # Specific recommendations based on feature importance
    top_feature = results['importance_df'].iloc[0]['feature']
    recommendations.append(f"• Highest priority: Monitor and track '{top_feature}' closely")

    return "\n".join(recommendations)


def generate_visualizations(results, data, output_dir):
    """Generate visualization plots for the report"""

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    importance_df = results['importance_df'].head(15)

    plt.subplot(2, 2, 1)
    bars = plt.barh(range(len(importance_df)), importance_df['VIP_score'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('VIP Score')
    plt.title('Top 15 Feature Importance (VIP Scores)')
    plt.gca().invert_yaxis()

    # Color bars by importance
    for i, bar in enumerate(bars):
        if importance_df.iloc[i]['VIP_score'] > 1.0:
            bar.set_color('red')
        elif importance_df.iloc[i]['VIP_score'] > 0.8:
            bar.set_color('orange')

    # 2. Performance Summary
    plt.subplot(2, 2, 2)
    hormones = list(results['r2_scores'].keys())
    r2_scores = [results['r2_scores'][h] for h in hormones]

    bars = plt.bar(hormones, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylim(0, 1)
    plt.ylabel('R² Score')
    plt.title('Model Performance by Hormone')

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    # 3. Data Completeness Heatmap
    plt.subplot(2, 2, 3)
    analysis_vars = results['predictors'] + results['targets']
    completeness_data = []

    for var in analysis_vars:
        non_missing = data[var].notna().sum()
        pct_complete = (non_missing / len(data)) * 100
        completeness_data.append(pct_complete)

    plt.barh(analysis_vars, completeness_data, color='lightblue')
    plt.xlabel('Data Completeness (%)')
    plt.title('Data Quality by Variable')
    plt.xlim(0, 100)

    # 4. Correlation Heatmap (Top 10 features)
    plt.subplot(2, 2, 4)
    top_features = results['importance_df'].head(10)['feature'].tolist()
    correlation_vars = top_features + results['targets']

    # Calculate correlations only for available data
    corr_data = data[correlation_vars].dropna()
    if len(corr_data) > 0:
        correlation_matrix = corr_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', cbar_kws={'label': 'Correlation'})
        plt.title('Top Features - Hormone Correlations')
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor correlation matrix',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Correlation Matrix (No Data)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pls_analysis_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Detailed Feature Importance Plot
    plt.figure(figsize=(10, 6))
    importance_df = results['importance_df']

    colors = ['red' if score > 1.0 else 'orange' if score > 0.8 else 'blue'
              for score in importance_df['VIP_score']]

    plt.bar(range(len(importance_df)), importance_df['VIP_score'], color=colors)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='VIP > 1 (Important)')
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='VIP > 0.8 (Moderate)')

    plt.xlabel('Features')
    plt.ylabel('VIP Score')
    plt.title('Feature Importance (VIP Scores) - All Features')
    plt.legend()
    plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_files(results, output_dir):
    """Generate CSV summary files"""

    # Feature importance table
    results['importance_df'].to_csv(f"{output_dir}/feature_importance.csv", index=False)

    # Performance summary
    performance_df = pd.DataFrame([
        {'hormone': hormone, 'r_squared': r2, 'interpretation': interpret_r2(r2)}
        for hormone, r2 in results['r2_scores'].items()
    ])
    performance_df.to_csv(f"{output_dir}/model_performance.csv", index=False)

    # Predictor statistics
    predictor_stats = []
    for predictor in results['predictors']:
        if predictor in results['X'].columns:
            stats = {
                'predictor': predictor,
                'mean': results['X'][predictor].mean(),
                'std': results['X'][predictor].std(),
                'vip_score': results['importance_df'].loc[
                    results['importance_df']['feature'] == predictor, 'VIP_score'
                ].values[0] if predictor in results['importance_df']['feature'].values else np.nan
            }
            predictor_stats.append(stats)

    pd.DataFrame(predictor_stats).to_csv(f"{output_dir}/predictor_statistics.csv", index=False)