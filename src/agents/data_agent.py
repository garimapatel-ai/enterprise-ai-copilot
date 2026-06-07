"""
Data Agent — autonomous data analysis, insight extraction, and executive summary generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    dataset_name: str
    shape: Tuple[int, int]
    summary_stats: Dict
    key_insights: List[str]
    anomalies: List[str]
    recommendations: List[str]
    executive_summary: str


class DataAgent:
    """
    Specialized agent for automated data analysis and insight generation.
    Performs statistical analysis, anomaly detection, and produces LLM-powered summaries.
    """

    def __init__(self, model: str = 'gpt-4o'):
        self.model = model
        self.analyses: List[AnalysisResult] = []
        logger.info(f"Data Agent initialized with model={model}")

    def analyze(self, df: pd.DataFrame, dataset_name: str = 'dataset') -> AnalysisResult:
        """Full automated analysis of a DataFrame."""
        logger.info(f"Analyzing dataset: {dataset_name} {df.shape}")

        summary = self._compute_summary(df)
        insights = self._extract_insights(df, summary)
        anomalies = self._detect_anomalies(df)
        recommendations = self._generate_recommendations(insights, anomalies)
        exec_summary = self._write_executive_summary(dataset_name, summary, insights)

        result = AnalysisResult(
            dataset_name=dataset_name,
            shape=df.shape,
            summary_stats=summary,
            key_insights=insights,
            anomalies=anomalies,
            recommendations=recommendations,
            executive_summary=exec_summary
        )
        self.analyses.append(result)
        return result

    def _compute_summary(self, df: pd.DataFrame) -> Dict:
        numeric = df.select_dtypes(include=[np.number])
        categorical = df.select_dtypes(include=['object', 'category'])
        return {
            'rows': len(df),
            'cols': len(df.columns),
            'missing_pct': round(df.isnull().mean().mean() * 100, 2),
            'numeric_cols': len(numeric.columns),
            'categorical_cols': len(categorical.columns),
            'numeric_stats': numeric.describe().to_dict() if not numeric.empty else {},
            'top_categories': {
                col: df[col].value_counts().head(3).to_dict()
                for col in categorical.columns[:5]
            }
        }

    def _extract_insights(self, df: pd.DataFrame, summary: Dict) -> List[str]:
        insights = []
        numeric = df.select_dtypes(include=[np.number])

        if summary['missing_pct'] > 10:
            insights.append(f"High missing data rate: {summary['missing_pct']:.1f}% — imputation recommended")
        elif summary['missing_pct'] == 0:
            insights.append("Dataset is complete — no missing values detected")

        for col in numeric.columns[:5]:
            skew = df[col].skew()
            if abs(skew) > 1.5:
                direction = 'right' if skew > 0 else 'left'
                insights.append(f"'{col}' is heavily {direction}-skewed (skew={skew:.2f}) — consider log transform")

        corr = numeric.corr() if len(numeric.columns) >= 2 else pd.DataFrame()
        if not corr.empty:
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.8:
                        insights.append(
                            f"Strong {'positive' if val > 0 else 'negative'} correlation "
                            f"between '{corr.columns[i]}' and '{corr.columns[j]}' (r={val:.2f})"
                        )

        if not insights:
            insights.append(f"Dataset has {summary['rows']:,} rows across {summary['cols']} features — clean structure")

        return insights[:8]

    def _detect_anomalies(self, df: pd.DataFrame) -> List[str]:
        anomalies = []
        numeric = df.select_dtypes(include=[np.number])

        for col in numeric.columns[:5]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            if outliers > 0:
                pct = outliers / len(df) * 100
                anomalies.append(f"'{col}': {outliers} outliers detected ({pct:.1f}% of data)")

        dupes = df.duplicated().sum()
        if dupes > 0:
            anomalies.append(f"{dupes} duplicate rows found")

        return anomalies[:5]

    def _generate_recommendations(self, insights: List[str], anomalies: List[str]) -> List[str]:
        recs = []
        if any('skewed' in i for i in insights):
            recs.append("Apply log or Box-Cox transformation to skewed features before modeling")
        if any('missing' in i for i in insights):
            recs.append("Use median imputation for numeric columns, mode for categorical")
        if any('correlation' in i for i in insights):
            recs.append("Consider PCA or feature selection to handle multicollinearity")
        if any('outlier' in a for a in anomalies):
            recs.append("Investigate outliers — apply IQR capping or robust scaler")
        if any('duplicate' in a for a in anomalies):
            recs.append("Remove duplicates with df.drop_duplicates() before training")
        if not recs:
            recs.append("Dataset appears ready for modeling — proceed with feature engineering")
        return recs

    def _write_executive_summary(self, name: str, summary: Dict, insights: List[str]) -> str:
        top_insight = insights[0] if insights else "No major issues detected."
        return (
            f"## Executive Summary — {name}\n\n"
            f"**Dataset Overview:** {summary['rows']:,} records × {summary['cols']} features. "
            f"Missing data: {summary['missing_pct']}%. "
            f"{summary['numeric_cols']} numeric and {summary['categorical_cols']} categorical columns.\n\n"
            f"**Key Finding:** {top_insight}\n\n"
            f"**Insights:** {len(insights)} patterns identified. "
            f"Recommend reviewing the {len(insights)} findings before proceeding to modeling."
        )

    def generate_report(self, result: AnalysisResult) -> str:
        lines = [
            result.executive_summary, "",
            "### Key Insights",
            *[f"- {i}" for i in result.key_insights], "",
            "### Anomalies Detected",
            *[f"⚠️  {a}" for a in result.anomalies] if result.anomalies else ["✅ No anomalies found"], "",
            "### Recommendations",
            *[f"→ {r}" for r in result.recommendations]
        ]
        return '\n'.join(lines)


if __name__ == '__main__':
    np.random.seed(42)
    df = pd.DataFrame({
        'revenue': np.random.lognormal(10, 1, 1000),
        'units_sold': np.random.randint(1, 500, 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'product': np.random.choice(['A', 'B', 'C'], 1000),
        'return_rate': np.random.beta(2, 10, 1000),
    })
    df.loc[np.random.choice(len(df), 50), 'revenue'] = np.nan

    agent = DataAgent()
    result = agent.analyze(df, 'Q4 Sales Data')
    print(agent.generate_report(result))
