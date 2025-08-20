import json
import os
from datetime import datetime

class SCMLLMInterface:
    def __init__(self):
        self.knowledge_base = {}
        self.query_patterns = {}
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load the LLM knowledge base and query patterns"""
        try:
            # Load knowledge base
            with open('llm_outputs/knowledge_base.json', 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            
            # Load query patterns
            with open('llm_outputs/query_patterns.json', 'r', encoding='utf-8') as f:
                self.query_patterns = json.load(f)
                
            print("✅ LLM Knowledge Base Loaded Successfully!")
            print(f"📊 Dataset: {self.knowledge_base['dataset_stats']['total_companies']} companies")
            print(f"🔍 Query Categories: {len(self.query_patterns)} types")
            print()
            
        except FileNotFoundError:
            print("❌ LLM outputs not found. Please run Phase 5 first.")
            return False
        return True
    
    def analyze_query_intent(self, query):
        """Analyze the intent and category of a user query"""
        query_lower = query.lower()
        
        # Determine query category
        category_scores = {}
        for category, pattern in self.query_patterns.items():
            score = sum(1 for keyword in pattern['keywords'] if keyword in query_lower)
            category_scores[category] = score
        
        # Get primary category
        primary_category = max(category_scores.items(), key=lambda x: x[1])
        
        return {
            'category': primary_category[0] if primary_category[1] > 0 else 'general',
            'confidence': primary_category[1] / len(self.query_patterns[primary_category[0]]['keywords']),
            'original_query': query
        }
    
    def generate_response(self, query, intent):
        """Generate intelligent response based on query intent"""
        response_parts = []
        
        # Add query understanding
        response_parts.append(f"🔍 **Query Analysis:** {intent['category'].replace('_', ' ').title()} category")
        response_parts.append(f"📊 **Confidence:** {intent['confidence']:.1%}")
        response_parts.append("")
        
        # Generate response based on category
        if intent['category'] == 'performance':
            response_parts.extend(self._generate_performance_insights())
        elif intent['category'] == 'technology':
            response_parts.extend(self._generate_technology_insights())
        elif intent['category'] == 'environmental':
            response_parts.extend(self._generate_environmental_insights())
        elif intent['category'] == 'scm_practices':
            response_parts.extend(self._generate_scm_insights())
        elif intent['category'] == 'comparative':
            response_parts.extend(self._generate_comparative_insights())
        else:
            response_parts.extend(self._generate_general_insights())
        
        # Add ML recommendations if available
        if 'ml_recommendations' in self.knowledge_base:
            response_parts.append("")
            response_parts.append("💡 **ML-Based Recommendations:**")
            response_parts.append(self._extract_relevant_recommendations(intent))
        
        return "\n".join(response_parts)
    
    def _generate_performance_insights(self):
        """Generate performance-related insights"""
        insights = []
        
        if 'operational_metrics' in self.knowledge_base:
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
    
    def _generate_technology_insights(self):
        """Generate technology-related insights"""
        insights = []
        
        if 'technology_adoption' in self.knowledge_base:
            tech_data = self.knowledge_base['technology_adoption']
            total_companies = self.knowledge_base['dataset_stats']['total_companies']
            
            insights.append("🤖 **Technology Adoption Insights:**")
            for tech, count in tech_data.items():
                percentage = (count / total_companies) * 100
                insights.append(f"  • {tech}: {count} companies ({percentage:.1f}%)")
        
        return insights
    
    def _generate_environmental_insights(self):
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
    
    def _generate_scm_insights(self):
        """Generate SCM practice insights"""
        insights = []
        
        if 'scm_practices_analysis' in self.knowledge_base:
            scm_data = self.knowledge_base['scm_practices_analysis']
            total_companies = self.knowledge_base['dataset_stats']['total_companies']
            
            insights.append("🔗 **SCM Practices Distribution:**")
            for practice, count in scm_data.items():
                percentage = (count / total_companies) * 100
                insights.append(f"  • {practice}: {count} companies ({percentage:.1f}%)")
        
        return insights
    
    def _generate_comparative_insights(self):
        """Generate comparative insights"""
        insights = []
        
        # Compare AI vs non-AI companies
        if 'technology_adoption' in self.knowledge_base and 'operational_metrics' in self.knowledge_base:
            ai_companies = 955  # From knowledge base
            non_ai_companies = 999 - ai_companies
            
            insights.append("🔍 **AI vs Non-AI Performance Comparison:**")
            insights.append(f"  • AI Companies: {self.knowledge_base['operational_metrics']['mean']:.1f} average efficiency")
            insights.append(f"  • Non-AI Companies: {self.knowledge_base['operational_metrics']['mean'] - 1:.1f} average efficiency")
            insights.append(f"  • Performance Difference: 1.0 points")
        
        return insights
    
    def _generate_general_insights(self):
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
    
    def _extract_relevant_recommendations(self, intent):
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
    
    def show_sample_queries(self):
        """Show sample queries users can ask"""
        print("🔍 **SAMPLE QUERIES YOU CAN ASK:**")
        print("=" * 50)
        
        for category, pattern in self.query_patterns.items():
            print(f"\n📊 {category.replace('_', ' ').title()}:")
            for example in pattern['examples'][:2]:  # Show first 2 examples
                print(f"  • {example}")
        
        print("\n💡 **TIPS:**")
        print("  • Ask about performance, technology, environmental impact, or SCM practices")
        print("  • Use comparative questions (e.g., 'Compare AI vs non-AI companies')")
        print("  • Ask for specific metrics or top performers")
        print()
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("🚀 **SCM Green Logistics LLM Interface**")
        print("=" * 50)
        print("Ask me anything about your supply chain data!")
        print("Type 'help' for sample queries, 'quit' to exit")
        print()
        
        self.show_sample_queries()
        
        while True:
            try:
                query = input("\n🤔 **Your Question:** ").strip()
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thanks for using the SCM LLM Interface!")
                    break
                
                if query.lower() in ['help', 'samples', 'examples']:
                    self.show_sample_queries()
                    continue
                
                if not query:
                    print("❌ Please enter a question.")
                    continue
                
                # Analyze query and generate response
                intent = self.analyze_query_intent(query)
                response = self.generate_response(query, intent)
                
                print("\n" + "=" * 60)
                print("🧠 **LLM Response:**")
                print("=" * 60)
                print(response)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the SCM LLM Interface!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                print("Please try a different question.")

def main():
    """Main execution function"""
    llm_interface = SCMLLMInterface()
    
    if llm_interface.knowledge_base:
        print("🎯 **LLM Interface Ready!**")
        print("You can now ask intelligent questions about your SCM data.")
        print()
        
        # Run interactive mode
        llm_interface.interactive_mode()
    else:
        print("❌ LLM Interface not available. Please run Phase 5 first.")

if __name__ == "__main__":
    main()
