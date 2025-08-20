import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
import json
import re
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class SCMLLMInsights:
    def __init__(self):
        self.df = None
        self.output_dir = 'llm_outputs'
        self.knowledge_base = {}
        self.query_patterns = {}
        self.insights_cache = {}
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory for LLM insights"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✅ Created output directory: {self.output_dir}")
    
    def load_data(self):
        """Load the cleaned SCM dataset and ML results"""
        print("🔄 Loading SCM dataset and ML results...")
        try:
            # Load main dataset
            self.df = pd.read_csv('scm_cleaned.csv')
            print(f"✅ SCM dataset loaded: {self.df.shape}")
            
            # Load ML results if available
            if os.path.exists('ml_outputs/ml_optimization_recommendations.txt'):
                with open('ml_outputs/ml_optimization_recommendations.txt', 'r', encoding='utf-8') as f:
                    ml_recommendations = f.read()
                self.knowledge_base['ml_recommendations'] = ml_recommendations
                print("✅ ML recommendations loaded")
            
            # Load KPI results if available
            if os.path.exists('kpi_outputs/kpi_analytics_report.txt'):
                with open('kpi_outputs/kpi_analytics_report.txt', 'r', encoding='utf-8') as f:
                    kpi_report = f.read()
                self.knowledge_base['kpi_report'] = kpi_report
                print("✅ KPI analytics report loaded")
            
            return True
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def build_knowledge_base(self):
        """Build comprehensive knowledge base from data"""
        print("\n" + "="*60)
        print("🧠 BUILDING KNOWLEDGE BASE")
        print("="*60)
        
        # 1. Dataset Statistics
        self.knowledge_base['dataset_stats'] = {
            'total_companies': len(self.df),
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        # 2. Key Metrics Summary
        if 'operational_efficiency_score' in self.df.columns:
            self.knowledge_base['operational_metrics'] = {
                'mean': self.df['operational_efficiency_score'].mean(),
                'median': self.df['operational_efficiency_score'].median(),
                'std': self.df['operational_efficiency_score'].std(),
                'min': self.df['operational_efficiency_score'].min(),
                'max': self.df['operational_efficiency_score'].max()
            }
        
        if 'environmental_impact_score' in self.df.columns:
            self.knowledge_base['environmental_metrics'] = {
                'mean': self.df['environmental_impact_score'].mean(),
                'median': self.df['environmental_impact_score'].median(),
                'std': self.df['environmental_impact_score'].std(),
                'min': self.df['environmental_impact_score'].min(),
                'max': self.df['environmental_impact_score'].max()
            }
        
        # 3. Categorical Analysis
        categorical_cols = ['scm_practices', 'supply_chain_agility', 'technology_utilized']
        for col in categorical_cols:
            if col in self.df.columns:
                self.knowledge_base[f'{col}_analysis'] = self.df[col].value_counts().to_dict()
        
        # 4. Technology Adoption Analysis
        if 'technology_utilized' in self.df.columns:
            tech_analysis = {}
            for tech in ['AI', 'Blockchain', 'Robotics', 'ERP']:
                tech_analysis[tech] = len(self.df[self.df['technology_utilized'].str.contains(tech, na=False)])
            self.knowledge_base['technology_adoption'] = tech_analysis
        
        # 5. Performance Rankings
        if 'operational_efficiency_score' in self.df.columns:
            top_performers = self.df.nlargest(10, 'operational_efficiency_score')[['company_name', 'operational_efficiency_score']]
            self.knowledge_base['top_operational_performers'] = top_performers.to_dict('records')
        
        if 'environmental_impact_score' in self.df.columns:
            top_environmental = self.df.nlargest(10, 'environmental_impact_score')[['company_name', 'environmental_impact_score']]
            self.knowledge_base['top_environmental_performers'] = top_environmental.to_dict('records')
        
        print("✅ Knowledge base built with comprehensive data insights")
        return True
    
    def create_query_patterns(self):
        """Create patterns for different types of queries"""
        print("\n" + "="*60)
        print("🔍 CREATING QUERY PATTERNS")
        print("="*60)
        
        # Performance queries
        self.query_patterns['performance'] = {
            'keywords': ['performance', 'efficiency', 'score', 'ranking', 'top', 'best', 'worst'],
            'examples': [
                "What are the top performing companies?",
                "Which companies have the highest efficiency scores?",
                "Show me the worst performing companies",
                "What's the average operational efficiency?"
            ]
        }
        
        # Technology queries
        self.query_patterns['technology'] = {
            'keywords': ['technology', 'AI', 'blockchain', 'robotics', 'ERP', 'adoption', 'digital'],
            'examples': [
                "How many companies use AI?",
                "What's the blockchain adoption rate?",
                "Which technologies are most popular?",
                "Show technology adoption trends"
            ]
        }
        
        # Environmental queries
        self.query_patterns['environmental'] = {
            'keywords': ['environmental', 'sustainability', 'green', 'eco', 'carbon', 'emission'],
            'examples': [
                "What are the environmental scores?",
                "Which companies are most sustainable?",
                "Show environmental performance trends",
                "What's the average environmental impact?"
            ]
        }
        
        # SCM practices queries
        self.query_patterns['scm_practices'] = {
            'keywords': ['SCM', 'supply chain', 'practice', 'methodology', 'approach', 'strategy'],
            'examples': [
                "What SCM practices are most common?",
                "Which practices lead to better performance?",
                "Show SCM practice distribution",
                "What are the different SCM approaches?"
            ]
        }
        
        # Comparative queries
        self.query_patterns['comparative'] = {
            'keywords': ['compare', 'versus', 'vs', 'difference', 'better', 'worse', 'relationship'],
            'examples': [
                "Compare AI vs non-AI companies",
                "What's the relationship between efficiency and environmental scores?",
                "How do different practices compare?",
                "Show performance by technology adoption"
            ]
        }
        
        print("✅ Query patterns created for 5 categories")
        return True
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and category of a user query"""
        query_lower = query.lower()
        
        # Determine query category
        category_scores = {}
        for category, pattern in self.query_patterns.items():
            score = sum(1 for keyword in pattern['keywords'] if keyword in query_lower)
            category_scores[category] = score
        
        # Get primary category
        primary_category = max(category_scores.items(), key=lambda x: x[1])
        
        # Extract specific metrics mentioned
        metrics_mentioned = []
        if 'efficiency' in query_lower or 'score' in query_lower:
            if 'operational' in query_lower:
                metrics_mentioned.append('operational_efficiency_score')
            if 'environmental' in query_lower:
                metrics_mentioned.append('environmental_impact_score')
            if 'cost' in query_lower:
                metrics_mentioned.append('cost_efficiency_index')
        
        # Extract comparison indicators
        is_comparison = any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference'])
        
        return {
            'category': primary_category[0] if primary_category[1] > 0 else 'general',
            'confidence': primary_category[1] / len(self.query_patterns[primary_category[0]]['keywords']),
            'metrics': metrics_mentioned,
            'is_comparison': is_comparison,
            'original_query': query
        }
    
    def generate_insight_response(self, query: str, intent: Dict[str, Any]) -> str:
        """Generate intelligent response based on query intent"""
        response_parts = []
        
        # Add query understanding
        response_parts.append(f"🔍 **Query Analysis:** {intent['category'].replace('_', ' ').title()} category")
        response_parts.append(f"📊 **Confidence:** {intent['confidence']:.1%}")
        response_parts.append("")
        
        # Generate response based on category
        if intent['category'] == 'performance':
            response_parts.extend(self._generate_performance_insights(query, intent))
        elif intent['category'] == 'technology':
            response_parts.extend(self._generate_technology_insights(query, intent))
        elif intent['category'] == 'environmental':
            response_parts.extend(self._generate_environmental_insights(query, intent))
        elif intent['category'] == 'scm_practices':
            response_parts.extend(self._generate_scm_insights(query, intent))
        elif intent['category'] == 'comparative':
            response_parts.extend(self._generate_comparative_insights(query, intent))
        else:
            response_parts.extend(self._generate_general_insights(query, intent))
        
        # Add recommendations if available
        if 'ml_recommendations' in self.knowledge_base:
            response_parts.append("")
            response_parts.append("💡 **ML-Based Recommendations:**")
            response_parts.append(self._extract_relevant_recommendations(intent))
        
        return "\n".join(response_parts)
    
    def _generate_performance_insights(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """Generate performance-related insights"""
        insights = []
        
        if 'operational_efficiency_score' in self.knowledge_base:
            metrics = self.knowledge_base['operational_metrics']
            insights.append("📈 **Operational Performance Insights:**")
            insights.append(f"  • Average Score: {metrics['mean']:.1f}")
            insights.append(f"  • Median Score: {metrics['median']:.1f}")
            insights.append(f"  • Score Range: {metrics['min']:.1f} - {metrics['max']:.1f}")
            insights.append(f"  • Standard Deviation: {metrics['std']:.1f}")
        
        if 'top_operational_performers' in self.knowledge_base:
            insights.append("")
            insights.append("🏆 **Top 5 Operational Performers:**")
            for i, company in enumerate(self.knowledge_base['top_operational_performers'][:5], 1):
                insights.append(f"  {i}. {company['company_name']}: {company['operational_efficiency_score']:.1f}")
        
        return insights
    
    def _generate_technology_insights(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """Generate technology-related insights"""
        insights = []
        
        if 'technology_adoption' in self.knowledge_base:
            tech_data = self.knowledge_base['technology_adoption']
            total_companies = self.knowledge_base['dataset_stats']['total_companies']
            
            insights.append("🤖 **Technology Adoption Insights:**")
            for tech, count in tech_data.items():
                percentage = (count / total_companies) * 100
                insights.append(f"  • {tech}: {count} companies ({percentage:.1f}%)")
        
        if 'technology_utilized_analysis' in self.knowledge_base:
            insights.append("")
            insights.append("📊 **Technology Distribution:**")
            tech_dist = self.knowledge_base['technology_utilized_analysis']
            for tech, count in list(tech_dist.items())[:5]:
                insights.append(f"  • {tech}: {count} companies")
        
        return insights
    
    def _generate_environmental_insights(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """Generate environmental-related insights"""
        insights = []
        
        if 'environmental_metrics' in self.knowledge_base:
            metrics = self.knowledge_base['environmental_metrics']
            insights.append("🌱 **Environmental Performance Insights:**")
            insights.append(f"  • Average Score: {metrics['mean']:.1f}")
            insights.append(f"  • Median Score: {metrics['median']:.1f}")
            insights.append(f"  • Score Range: {metrics['min']:.1f} - {metrics['max']:.1f}")
            insights.append(f"  • Standard Deviation: {metrics['std']:.1f}")
        
        if 'top_environmental_performers' in self.knowledge_base:
            insights.append("")
            insights.append("🌿 **Top 5 Environmental Performers:**")
            for i, company in enumerate(self.knowledge_base['top_environmental_performers'][:5], 1):
                insights.append(f"  {i}. {company['company_name']}: {company['environmental_impact_score']:.1f}")
        
        return insights
    
    def _generate_scm_insights(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """Generate SCM practice insights"""
        insights = []
        
        if 'scm_practices_analysis' in self.knowledge_base:
            scm_data = self.knowledge_base['scm_practices_analysis']
            total_companies = self.knowledge_base['dataset_stats']['total_companies']
            
            insights.append("🔗 **SCM Practices Distribution:**")
            for practice, count in scm_data.items():
                percentage = (count / total_companies) * 100
                insights.append(f"  • {practice}: {count} companies ({percentage:.1f}%)")
        
        if 'supply_chain_agility_analysis' in self.knowledge_base:
            insights.append("")
            insights.append("⚡ **Supply Chain Agility:**")
            agility_data = self.knowledge_base['supply_chain_agility_analysis']
            for level, count in agility_data.items():
                percentage = (count / total_companies) * 100
                insights.append(f"  • {level}: {count} companies ({percentage:.1f}%)")
        
        return insights
    
    def _generate_comparative_insights(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """Generate comparative insights"""
        insights = []
        
        # Compare AI vs non-AI companies
        if 'technology_adoption' in self.knowledge_base and 'operational_metrics' in self.knowledge_base:
            ai_companies = self.df[self.df['technology_utilized'].str.contains('AI', na=False)]
            non_ai_companies = self.df[~self.df['technology_utilized'].str.contains('AI', na=False)]
            
            if len(ai_companies) > 0 and len(non_ai_companies) > 0:
                ai_avg = ai_companies['operational_efficiency_score'].mean()
                non_ai_avg = non_ai_companies['operational_efficiency_score'].mean()
                
                insights.append("🔍 **AI vs Non-AI Performance Comparison:**")
                insights.append(f"  • AI Companies: {ai_avg:.1f} average efficiency")
                insights.append(f"  • Non-AI Companies: {non_ai_avg:.1f} average efficiency")
                insights.append(f"  • Performance Difference: {ai_avg - non_ai_avg:.1f} points")
        
        # Compare SCM practices
        if 'scm_practices_analysis' in self.knowledge_base and 'operational_metrics' in self.knowledge_base:
            insights.append("")
            insights.append("📊 **SCM Practice Performance Comparison:**")
            practice_performance = self.df.groupby('scm_practices')['operational_efficiency_score'].mean().sort_values(ascending=False)
            for practice, score in practice_performance.head(3).items():
                insights.append(f"  • {practice}: {score:.1f} average efficiency")
        
        return insights
    
    def _generate_general_insights(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """Generate general insights"""
        insights = []
        
        insights.append("📋 **General Dataset Overview:**")
        insights.append(f"  • Total Companies Analyzed: {self.knowledge_base['dataset_stats']['total_companies']}")
        insights.append(f"  • Data Dimensions: {len(self.knowledge_base['dataset_stats']['columns'])} columns")
        
        if 'operational_metrics' in self.knowledge_base:
            insights.append(f"  • Operational Efficiency Range: {self.knowledge_base['operational_metrics']['min']:.1f} - {self.knowledge_base['operational_metrics']['max']:.1f}")
        
        if 'environmental_metrics' in self.knowledge_base:
            insights.append(f"  • Environmental Impact Range: {self.knowledge_base['environmental_metrics']['min']:.1f} - {self.knowledge_base['environmental_metrics']['max']:.1f}")
        
        return insights
    
    def _extract_relevant_recommendations(self, intent: Dict[str, Any]) -> str:
        """Extract relevant ML recommendations based on query intent"""
        if 'ml_recommendations' not in self.knowledge_base:
            return "No ML recommendations available."
        
        recommendations = self.knowledge_base['ml_recommendations']
        
        # Extract relevant sections based on intent
        relevant_parts = []
        
        if intent['category'] == 'technology':
            if 'AI' in recommendations or 'Blockchain' in recommendations:
                relevant_parts.append("• Prioritize AI and Blockchain adoption for operational efficiency")
        
        if intent['category'] == 'environmental':
            if 'sustainable' in recommendations or 'environmental' in recommendations:
                relevant_parts.append("• Focus on sustainable SCM practices for environmental impact")
        
        if intent['category'] == 'performance':
            if 'efficiency' in recommendations or 'optimization' in recommendations:
                relevant_parts.append("• Implement technology-driven supply chain optimization")
        
        if not relevant_parts:
            relevant_parts.append("• Develop data-driven decision making frameworks")
            relevant_parts.append("• Implement continuous optimization and monitoring")
        
        return "\n".join(relevant_parts)
    
    def create_insight_dashboard(self):
        """Create comprehensive insight dashboard"""
        print("\n" + "="*60)
        print("📊 CREATING INSIGHT DASHBOARD")
        print("="*60)
        
        # Create summary visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SCM Green Logistics - LLM Insights Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Technology Adoption
        if 'technology_adoption' in self.knowledge_base:
            tech_data = self.knowledge_base['technology_adoption']
            tech_names = list(tech_data.keys())
            tech_counts = list(tech_data.values())
            
            axes[0, 0].bar(tech_names, tech_counts, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Technology Adoption Rates', fontweight='bold')
            axes[0, 0].set_ylabel('Number of Companies')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. SCM Practices Distribution
        if 'scm_practices_analysis' in self.knowledge_base:
            scm_data = self.knowledge_base['scm_practices_analysis']
            scm_names = list(scm_data.keys())[:5]  # Top 5
            scm_counts = list(scm_data.values())[:5]
            
            axes[0, 1].pie(scm_counts, labels=scm_names, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('SCM Practices Distribution', fontweight='bold')
        
        # 3. Performance Distribution
        if 'operational_metrics' in self.knowledge_base:
            scores = self.df['operational_efficiency_score'].dropna()
            axes[1, 0].hist(scores, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('Operational Efficiency Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Efficiency Score')
            axes[1, 0].set_ylabel('Number of Companies')
            axes[1, 0].axvline(scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.1f}')
            axes[1, 0].legend()
        
        # 4. Environmental vs Operational Correlation
        if 'operational_efficiency_score' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            axes[1, 1].scatter(self.df['operational_efficiency_score'], self.df['environmental_impact_score'], alpha=0.6, s=30)
            axes[1, 1].set_xlabel('Operational Efficiency Score')
            axes[1, 1].set_ylabel('Environmental Impact Score')
            axes[1, 1].set_title('Operational vs Environmental Performance', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/llm_insights_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ LLM insights dashboard visualization saved")
    
    def save_knowledge_base(self):
        """Save the knowledge base for future use"""
        print("\n" + "="*60)
        print("💾 SAVING KNOWLEDGE BASE")
        print("="*60)
        
        # Save as JSON
        with open(f'{self.output_dir}/knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, default=str)
        
        # Save query patterns
        with open(f'{self.output_dir}/query_patterns.json', 'w', encoding='utf-8') as f:
            json.dump(self.query_patterns, f, indent=2)
        
        print("✅ Knowledge base saved as JSON files")
    
    def generate_sample_queries_report(self):
        """Generate a report with sample queries and responses"""
        print("\n" + "="*60)
        print("📝 GENERATING SAMPLE QUERIES REPORT")
        print("="*60)
        
        sample_queries = [
            "What are the top performing companies?",
            "How many companies use AI technology?",
            "Which companies are most environmentally friendly?",
            "What SCM practices are most common?",
            "Compare AI vs non-AI company performance",
            "What's the average operational efficiency?",
            "Show me blockchain adoption rates",
            "Which practices lead to better environmental scores?"
        ]
        
        report = []
        report.append("=" * 80)
        report.append("SCM GREEN LOGISTICS - LLM INSIGHTS SAMPLE QUERIES")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for i, query in enumerate(sample_queries, 1):
            report.append(f"Query {i}: {query}")
            intent = self.analyze_query_intent(query)
            response = self.generate_insight_response(query, intent)
            report.append("Response:")
            report.append(response)
            report.append("-" * 80)
            report.append("")
        
        # Save report
        with open(f'{self.output_dir}/sample_queries_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("✅ Sample queries report saved")
        
        # Print first few queries as example
        print("\n📋 Sample Queries (first 3):")
        for i, query in enumerate(sample_queries[:3], 1):
            print(f"  {i}. {query}")
    
    def run_full_llm_integration(self):
        """Run the complete LLM integration pipeline"""
        print("🚀 Starting SCM Green Logistics LLM Integration")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Build knowledge base
        if not self.build_knowledge_base():
            return False
        
        # Create query patterns
        if not self.create_query_patterns():
            return False
        
        # Create insight dashboard
        self.create_insight_dashboard()
        
        # Save knowledge base
        self.save_knowledge_base()
        
        # Generate sample queries report
        self.generate_sample_queries_report()
        
        print("\n" + "=" * 70)
        print("✅ LLM INTEGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 All outputs saved in: {self.output_dir}/")
        print("🧠 Knowledge base built and saved")
        print("🔍 Query patterns created")
        print("📊 Insight dashboard generated")
        print("📝 Sample queries report created")
        print("🎯 Ready for Phase 6: Dashboard & Visualization")
        
        return True

def main():
    """Main execution function"""
    llm_integration = SCMLLMInsights()
    success = llm_integration.run_full_llm_integration()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Review LLM outputs in llm_outputs/ folder")
        print("2. Examine knowledge base and query patterns")
        print("3. Test sample queries and responses")
        print("4. Proceed to Phase 6: Dashboard & Visualization")
    else:
        print("\n❌ LLM integration failed. Please check the errors above.")

if __name__ == "__main__":
    main()
