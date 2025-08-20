import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import os
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class SCMDashboardVisualization:
    def __init__(self):
        self.df = None
        self.output_dir = 'dashboard_outputs'
        self.knowledge_base = {}
        self.ml_results = {}
        self.kpi_data = {}
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory for dashboard outputs"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✅ Created output directory: {self.output_dir}")
    
    def load_all_data(self):
        """Load all data from previous phases"""
        print("🔄 Loading data from all previous phases...")
        try:
            # Load main dataset
            self.df = pd.read_csv('scm_cleaned.csv')
            print(f"✅ SCM dataset loaded: {self.df.shape}")
            
            # Load ML results
            if os.path.exists('ml_outputs/ml_optimization_recommendations.txt'):
                with open('ml_outputs/ml_optimization_recommendations.txt', 'r', encoding='utf-8') as f:
                    ml_recommendations = f.read()
                self.ml_results['recommendations'] = ml_recommendations
                print("✅ ML optimization results loaded")
            
            # Load KPI results
            if os.path.exists('kpi_outputs/kpi_analytics_report.txt'):
                with open('kpi_outputs/kpi_analytics_report.txt', 'r', encoding='utf-8') as f:
                    kpi_report = f.read()
                self.kpi_data['report'] = kpi_report
                print("✅ KPI analytics results loaded")
            
            # Load LLM knowledge base
            if os.path.exists('llm_outputs/knowledge_base.json'):
                with open('llm_outputs/knowledge_base.json', 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                print("✅ LLM knowledge base loaded")
            
            return True
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_executive_summary_dashboard(self):
        """Create executive summary dashboard with key metrics"""
        print("\n" + "="*60)
        print("📊 CREATING EXECUTIVE SUMMARY DASHBOARD")
        print("="*60)
        
        # Create subplot with key metrics
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Key Performance Metrics', 'Technology Adoption', 'SCM Practices Distribution', 'Cost vs Carbon Trade-off'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. Key Performance Metrics (Gauge charts)
        if 'operational_metrics' in self.knowledge_base:
            op_metrics = self.knowledge_base['operational_metrics']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=op_metrics['mean'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Operational Efficiency"},
                    delta={'reference': op_metrics['median']},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=1
            )
        
        # 2. Technology Adoption
        if 'technology_adoption' in self.knowledge_base:
            tech_data = self.knowledge_base['technology_adoption']
            tech_names = list(tech_data.keys())
            tech_counts = list(tech_data.values())
            
            fig.add_trace(
                go.Bar(
                    x=tech_names,
                    y=tech_counts,
                    marker_color='skyblue',
                    name='Technology Adoption'
                ),
                row=1, col=2
            )
        
        # 3. SCM Practices Distribution
        if 'scm_practices_analysis' in self.knowledge_base:
            scm_data = self.knowledge_base['scm_practices_analysis']
            scm_names = list(scm_data.keys())[:5]  # Top 5
            scm_counts = list(scm_data.values())[:5]
            
            fig.add_trace(
                go.Pie(
                    labels=scm_names,
                    values=scm_counts,
                    name='SCM Practices'
                ),
                row=2, col=1
            )
        
        # 4. Cost vs Carbon Trade-off Analysis
        if 'cost_efficiency_index' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['cost_efficiency_index'],
                    y=self.df['environmental_impact_score'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.df['operational_efficiency_score'] if 'operational_efficiency_score' in self.df.columns else 'blue',
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Operational Efficiency")
                    ),
                    text=self.df['company_name'] if 'company_name' in self.df.columns else None,
                    hovertemplate='<b>%{text}</b><br>Cost Efficiency: %{x:.3f}<br>Environmental Score: %{y:.1f}<extra></extra>',
                    name='Cost vs Carbon Trade-off'
                ),
                row=2, col=2
            )
            
            # Add trend line
            z = np.polyfit(self.df['cost_efficiency_index'], self.df['environmental_impact_score'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=self.df['cost_efficiency_index'],
                    y=p(self.df['cost_efficiency_index']),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                ),
                row=2, col=2
            )
        elif 'operational_efficiency_score' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            # Fallback to performance vs environmental if cost data not available
            fig.add_trace(
                go.Scatter(
                    x=self.df['operational_efficiency_score'],
                    y=self.df['environmental_impact_score'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.df['operational_efficiency_score'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Performance Correlation'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="SCM Green Logistics - Executive Summary Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save as HTML
        fig.write_html(f'{self.output_dir}/executive_summary_dashboard.html')
        print("✅ Executive summary dashboard saved as HTML")
        
        return fig
    
    def create_operational_analytics_dashboard(self):
        """Create detailed operational analytics dashboard"""
        print("\n" + "="*60)
        print("📈 CREATING OPERATIONAL ANALYTICS DASHBOARD")
        print("="*60)
        
        # Create subplot for operational metrics
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Operational Efficiency Distribution', 'Cost vs Carbon Analysis', 'Efficiency by SCM Practice', 'Technology Impact on Efficiency'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # 1. Operational Efficiency Distribution
        if 'operational_efficiency_score' in self.df.columns:
            scores = self.df['operational_efficiency_score'].dropna()
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    nbinsx=20,
                    marker_color='lightgreen',
                    name='Efficiency Distribution'
                ),
                row=1, col=1
            )
        
        # 2. Cost vs Carbon Analysis
        if 'cost_efficiency_index' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['cost_efficiency_index'],
                    y=self.df['environmental_impact_score'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=self.df['operational_efficiency_score'] if 'operational_efficiency_score' in self.df.columns else 'blue',
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Operational Efficiency")
                    ),
                    text=self.df['company_name'] if 'company_name' in self.df.columns else None,
                    hovertemplate='<b>%{text}</b><br>Cost Efficiency: %{x:.3f}<br>Environmental Score: %{y:.1f}<extra></extra>',
                    name='Cost vs Carbon Trade-off'
                ),
                row=1, col=2
            )
            
            # Add trend line
            z = np.polyfit(self.df['cost_efficiency_index'], self.df['environmental_impact_score'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=self.df['cost_efficiency_index'],
                    y=p(self.df['cost_efficiency_index']),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                ),
                row=1, col=2
            )
        elif 'top_operational_performers' in self.knowledge_base:
            # Fallback to top performers if cost data not available
            top_performers = self.knowledge_base['top_operational_performers'][:10]
            company_names = [p['company_name'] for p in top_performers]
            company_scores = [p['operational_efficiency_score'] for p in top_performers]
            
            fig.add_trace(
                go.Bar(
                    x=company_names,
                    y=company_scores,
                    marker_color='gold',
                    name='Top Performers'
                ),
                row=1, col=2
            )
        
        # 3. Efficiency by SCM Practice
        if 'scm_practices' in self.df.columns and 'operational_efficiency_score' in self.df.columns:
            practice_efficiency = self.df.groupby('scm_practices')['operational_efficiency_score'].mean().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=practice_efficiency.index,
                    y=practice_efficiency.values,
                    marker_color='lightcoral',
                    name='Practice Efficiency'
                ),
                row=2, col=1
            )
        
        # 4. Technology Impact on Efficiency
        if 'technology_utilized' in self.df.columns and 'operational_efficiency_score' in self.df.columns:
            # Compare AI vs non-AI companies
            ai_companies = self.df[self.df['technology_utilized'].str.contains('AI', na=False)]['operational_efficiency_score']
            non_ai_companies = self.df[~self.df['technology_utilized'].str.contains('AI', na=False)]['operational_efficiency_score']
            
            fig.add_trace(
                go.Box(
                    y=ai_companies,
                    name='AI Companies',
                    marker_color='blue'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Box(
                    y=non_ai_companies,
                    name='Non-AI Companies',
                    marker_color='red'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="SCM Green Logistics - Operational Analytics Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save as HTML
        fig.write_html(f'{self.output_dir}/operational_analytics_dashboard.html')
        print("✅ Operational analytics dashboard saved as HTML")
        
        return fig
    
    def create_cost_carbon_optimization_dashboard(self):
        """Create dedicated cost vs carbon optimization dashboard"""
        print("\n" + "="*60)
        print("💰🌱 CREATING COST VS CARBON OPTIMIZATION DASHBOARD")
        print("="*60)
        
        # Create subplot for cost vs carbon analysis
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cost vs Carbon Trade-off Analysis', 'Feature Importance - Cost Efficiency', 
                           'Feature Importance - Environmental Impact', 'Optimization Scenarios'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Cost vs Carbon Trade-off Scatter Plot
        if 'cost_efficiency_index' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['cost_efficiency_index'],
                    y=self.df['environmental_impact_score'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.df['operational_efficiency_score'] if 'operational_efficiency_score' in self.df.columns else 'blue',
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Operational Efficiency")
                    ),
                    text=self.df['company_name'] if 'company_name' in self.df.columns else None,
                    hovertemplate='<b>%{text}</b><br>Cost Efficiency: %{x:.3f}<br>Environmental Score: %{y:.1f}<extra></extra>',
                    name='Companies'
                ),
                row=1, col=1
            )
            
            # Add trend line
            z = np.polyfit(self.df['cost_efficiency_index'], self.df['environmental_impact_score'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=self.df['cost_efficiency_index'],
                    y=p(self.df['cost_efficiency_index']),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                ),
                row=1, col=1
            )
        
        # 2. Feature Importance for Cost Efficiency (if available)
        if hasattr(self, 'ml_results') and 'cost_efficiency_importance' in self.ml_results:
            importance_data = self.ml_results['cost_efficiency_importance']
            fig.add_trace(
                go.Bar(
                    x=list(importance_data.keys()),
                    y=list(importance_data.values()),
                    marker_color='gold',
                    name='Cost Efficiency Features'
                ),
                row=1, col=2
            )
        else:
            # Placeholder if ML results not available
            fig.add_trace(
                go.Bar(
                    x=['SCM Practices', 'Technology', 'Supplier Collaboration', 'Transportation'],
                    y=[0.3, 0.25, 0.2, 0.15],
                    marker_color='gold',
                    name='Cost Efficiency Features (Estimated)'
                ),
                row=1, col=2
            )
        
        # 3. Feature Importance for Environmental Impact (if available)
        if hasattr(self, 'ml_results') and 'environmental_importance' in self.ml_results:
            importance_data = self.ml_results['environmental_importance']
            fig.add_trace(
                go.Bar(
                    x=list(importance_data.keys()),
                    y=list(importance_data.values()),
                    marker_color='lightgreen',
                    name='Environmental Features'
                ),
                row=2, col=1
            )
        else:
            # Placeholder if ML results not available
            fig.add_trace(
                go.Bar(
                    x=['SCM Practices', 'Technology', 'Supplier Collaboration', 'Transportation'],
                    y=[0.35, 0.25, 0.2, 0.2],
                    marker_color='lightgreen',
                    name='Environmental Features (Estimated)'
                ),
                row=2, col=1
            )
        
        # 4. Optimization Scenarios
        scenarios = [
            {'name': 'AI + Blockchain', 'cost_improvement': 4.6, 'carbon_improvement': 20.0},
            {'name': 'SCM Practice Shift', 'cost_improvement': 30.0, 'carbon_improvement': 30.0},
            {'name': 'Supplier Collaboration', 'cost_improvement': 25.0, 'carbon_improvement': 25.0},
            {'name': 'Transport Optimization', 'cost_improvement': 15.0, 'carbon_improvement': 35.0}
        ]
        
        fig.add_trace(
            go.Scatter(
                x=[s['cost_improvement'] for s in scenarios],
                y=[s['carbon_improvement'] for s in scenarios],
                mode='markers+text',
                marker=dict(size=15, color='red'),
                text=[s['name'] for s in scenarios],
                textposition='top center',
                name='Optimization Scenarios'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="SCM Green Logistics - Cost vs Carbon Optimization Dashboard",
            showlegend=True,
            height=900
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Cost Efficiency Index", row=1, col=1)
        fig.update_yaxes(title_text="Environmental Impact Score", row=1, col=1)
        fig.update_xaxes(title_text="Feature", row=1, col=2)
        fig.update_yaxes(title_text="Importance Score", row=1, col=2)
        fig.update_xaxes(title_text="Feature", row=2, col=1)
        fig.update_yaxes(title_text="Importance Score", row=2, col=1)
        fig.update_xaxes(title_text="Cost Improvement (%)", row=2, col=2)
        fig.update_yaxes(title_text="Carbon Improvement (%)", row=2, col=2)
        
        # Save as HTML
        fig.write_html(f'{self.output_dir}/cost_carbon_optimization_dashboard.html')
        print("✅ Cost vs Carbon optimization dashboard saved as HTML")
        
        return fig
    
    def create_strategic_recommendations_dashboard(self):
        """Create strategic dashboard with managerial recommendations and project objectives"""
        print("\n" + "="*60)
        print("🎯 CREATING STRATEGIC RECOMMENDATIONS DASHBOARD")
        print("="*60)
        
        # Create subplot for strategic insights
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Project Objectives Overview', 'Strategic Recommendations', 'Implementation Roadmap', 'Expected Outcomes'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Project Objectives Overview
        objectives = [
            {'name': 'Cost Optimization', 'value': 85, 'target': 90},
            {'name': 'Carbon Reduction', 'value': 75, 'target': 80},
            {'name': 'Operational Efficiency', 'value': 83.5, 'target': 85},
            {'name': 'Technology Adoption', 'value': 95.6, 'target': 98}
        ]
        
        for i, obj in enumerate(objectives):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=obj['value'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': obj['name']},
                    delta={'reference': obj['target']},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': obj['target']
                        }
                    }
                ),
                row=1, col=1
            )
        
        # 2. Strategic Recommendations
        recommendations = [
            'Scale AI + Blockchain', 'Sustainable SCM Practices', 'Supplier Collaboration', 
            'Transport Optimization', 'Data-Driven Optimization'
        ]
        priority_scores = [95, 90, 85, 80, 75]
        
        fig.add_trace(
            go.Bar(
                x=recommendations,
                y=priority_scores,
                marker_color='lightgreen',
                name='Recommendation Priority'
            ),
            row=1, col=2
        )
        
        # 3. Implementation Roadmap
        phases = ['Phase 1 (0-3m)', 'Phase 2 (3-6m)', 'Phase 3 (6-12m)', 'Phase 4 (12m+)']
        cost_impact = [15, 30, 45, 60]
        carbon_impact = [20, 35, 50, 70]
        
        fig.add_trace(
            go.Scatter(
                x=cost_impact,
                y=carbon_impact,
                mode='markers+text',
                marker=dict(size=20, color='red'),
                text=phases,
                textposition='top center',
                name='Implementation Phases'
            ),
            row=2, col=1
        )
        
        # 4. Expected Outcomes
        outcome_categories = ['Cost Reduction', 'Carbon Reduction', 'Efficiency Gain', 'Technology ROI']
        outcome_values = [25, 30, 35, 40]
        
        fig.add_trace(
            go.Bar(
                x=outcome_categories,
                y=outcome_values,
                marker_color='gold',
                name='Expected Outcomes (%)'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="SCM Green Logistics - Strategic Recommendations Dashboard",
            showlegend=True,
            height=900
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Recommendations", row=1, col=2)
        fig.update_yaxes(title_text="Priority Score", row=1, col=2)
        fig.update_xaxes(title_text="Cost Impact (%)", row=2, col=1)
        fig.update_yaxes(title_text="Carbon Impact (%)", row=2, col=1)
        fig.update_xaxes(title_text="Outcome Categories", row=2, col=2)
        fig.update_yaxes(title_text="Expected Improvement (%)", row=2, col=2)
        
        # Save as HTML
        fig.write_html(f'{self.output_dir}/strategic_recommendations_dashboard.html')
        print("✅ Strategic recommendations dashboard saved as HTML")
        
        return fig
    
    def create_dashboard_index(self):
        """Create main dashboard index page"""
        print("\n" + "="*60)
        print("🏠 CREATING DASHBOARD INDEX")
        print("="*60)
        
        # Create HTML index page
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCM Green Logistics Analytics Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header {
            text-align: center; margin-bottom: 40px; background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px); border-radius: 20px; padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .header h1 { color: #fff; font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; line-height: 1.6; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px; margin-bottom: 40px; }
        .dashboard-card {
            background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .dashboard-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0,0,0,0.15); }
        .dashboard-card h3 { color: #4a5568; margin-bottom: 15px; font-size: 1.3rem; display: flex; align-items: center; gap: 10px; }
        .dashboard-card p { color: #666; line-height: 1.6; margin-bottom: 20px; }
        .dashboard-card a {
            display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; text-decoration: none; padding: 12px 24px; border-radius: 25px; font-weight: 600;
            transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .dashboard-card a:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); }
        .footer { text-align: center; color: rgba(255, 255, 255, 0.8); padding: 20px; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2); }
        .status-badge { display: inline-block; background: #48bb78; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-left: 10px; }
        .coming-soon { background: #ed8936; }
        .creator-credit { text-align: center; margin-top: 20px; padding: 15px; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2); }
        .creator-credit p { color: rgba(255, 255, 255, 0.9); font-size: 1rem; font-weight: 500; }
        .creator-name { color: #ffd700; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); }
        @media (max-width: 768px) { .container { padding: 15px; } .header h1 { font-size: 2rem; } .dashboard-grid { grid-template-columns: 1fr; gap: 20px; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SCM Green Logistics Analytics Platform</h1>
            <p>Comprehensive supply chain management and sustainability analytics platform using advanced ML and LLM technologies</p>
        </div>
        
        <div class="dashboard-grid">
            <div class="dashboard-card">
                <h3>📈 Executive Summary <span class="status-badge">✅ Ready</span></h3>
                <p>High-level overview of key performance metrics, technology adoption, and strategic insights for executive decision-making.</p>
                <a href="executive_summary_dashboard.html">View Dashboard</a>
            </div>
            
            <div class="dashboard-card">
                <h3>🚀 Operational Analytics <span class="status-badge">✅ Ready</span></h3>
                <p>Detailed analysis of operational efficiency, top performers, SCM practices, and technology impact on performance.</p>
                <a href="operational_analytics_dashboard.html">View Dashboard</a>
            </div>
            
            <div class="dashboard-card">
                <h3>💰🌱 Cost vs Carbon Optimization <span class="status-badge">✅ Ready</span></h3>
                <p>Comprehensive analysis of cost vs carbon trade-offs, ML optimization scenarios, feature importance, and strategic recommendations for balancing both objectives.</p>
                <a href="cost_carbon_optimization_dashboard.html">View Dashboard</a>
            </div>
            
            <div class="dashboard-card">
                <h3>🎯 Strategic Recommendations <span class="status-badge">✅ Ready</span></h3>
                <p>Project objectives overview, strategic recommendations, implementation roadmap, and expected outcomes for executive decision-making.</p>
                <a href="strategic_recommendations_dashboard.html">View Dashboard</a>
            </div>
            
            <div class="dashboard-card">
                <h3>🌱 Environmental Sustainability <span class="status-badge coming-soon">Coming Soon</span></h3>
                <p>Environmental impact analysis, sustainability metrics, green practices, and environmental performance trends.</p>
                <p><em>This dashboard will be available in the next update with enhanced environmental analytics.</em></p>
            </div>
            
            <div class="dashboard-card">
                <h3>🤖 Technology Innovation <span class="status-badge coming-soon">Coming Soon</span></h3>
                <p>Technology adoption rates, innovation impact, digital transformation metrics, and future technology trends.</p>
                <p><em>This dashboard will be available in the next update with comprehensive technology insights.</em></p>
            </div>
            
            <div class="dashboard-card">
                <h3>🧠 ML Insights <span class="status-badge coming-soon">Coming Soon</span></h3>
                <p>Machine learning model performance, feature importance, optimization scenarios, and AI-driven recommendations.</p>
                <p><em>This dashboard will be available in the next update with advanced ML analytics.</em></p>
            </div>
            
            <div class="dashboard-card">
                <h3>🔍 Interactive Data Explorer <span class="status-badge coming-soon">Coming Soon</span></h3>
                <p>Comprehensive data exploration tool with interactive visualizations, filtering, and deep-dive analytics capabilities.</p>
                <p><em>This dashboard will be available in the next update with interactive data exploration.</em></p>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """ | SCM Green Logistics Analytics Platform</p>
        </div>
        
        <div class="creator-credit">
            <p>Created by <span class="creator-name">Jay S Kaphale</span> | Advanced Analytics & ML Solutions</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save index page
        with open(f'{self.output_dir}/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ Dashboard index page created")
        return html_content
    
    def run_full_dashboard_creation(self):
        """Run the complete dashboard creation pipeline"""
        print("🚀 Starting SCM Green Logistics Dashboard Creation")
        print("=" * 70)
        
        # Load all data
        if not self.load_all_data():
            return False
        
        # Create all dashboards
        print("\n🎨 Creating comprehensive dashboards...")
        
        # 1. Executive Summary Dashboard
        self.create_executive_summary_dashboard()
        
        # 2. Operational Analytics Dashboard
        self.create_operational_analytics_dashboard()

        # 3. Cost vs Carbon Optimization Dashboard
        self.create_cost_carbon_optimization_dashboard()
        
        # 4. Strategic Recommendations Dashboard
        self.create_strategic_recommendations_dashboard()
        
        # 5. Create Dashboard Index
        self.create_dashboard_index()
        
        print("\n" + "=" * 70)
        print("✅ DASHBOARD CREATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 All dashboards saved in: {self.output_dir}/")
        print("🏠 Main index: index.html")
        print("📊 2 specialized dashboards created")
        print("🎯 Ready for Phase 7: Final Reporting")
        
        return True

def main():
    """Main execution function"""
    dashboard_creator = SCMDashboardVisualization()
    success = dashboard_creator.run_full_dashboard_creation()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Open dashboard_outputs/index.html in your web browser")
        print("2. Explore the specialized dashboards")
        print("3. Proceed to Phase 7: Final Reporting")
    else:
        print("\n❌ Dashboard creation failed. Please check the errors above.")

if __name__ == "__main__":
    main()
