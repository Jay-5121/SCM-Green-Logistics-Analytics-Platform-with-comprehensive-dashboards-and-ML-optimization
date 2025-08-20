import json
import os
import pandas as pd
from datetime import datetime

class SmartSCMLLMInterface:
    def __init__(self):
        self.knowledge_base = {}
        self.query_patterns = {}
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the LLM knowledge base, query patterns, and actual dataset"""
        try:
            # Load knowledge base
            with open('llm_outputs/knowledge_base.json', 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            
            # Load query patterns
            with open('llm_outputs/query_patterns.json', 'r', encoding='utf-8') as f:
                self.query_patterns = json.load(f)
            
            # Load actual dataset for real answers
            self.df = pd.read_csv('scm_cleaned.csv')
                
            print("✅ Smart LLM Interface Loaded Successfully!")
            print(f"📊 Dataset: {len(self.df)} companies with {len(self.df.columns)} dimensions")
            print(f"🔍 Query Categories: {len(self.query_patterns)} types")
            print()
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
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
    
    def generate_smart_response(self, query, intent):
        """Generate intelligent response based on actual data analysis"""
        response_parts = []
        
        # Add query understanding
        response_parts.append(f"🔍 **Query Analysis:** {intent['category'].replace('_', ' ').title()} category")
        response_parts.append(f"📊 **Confidence:** {intent['confidence']:.1%}")
        response_parts.append("")
        
        # Generate smart response based on actual data
        if intent['category'] == 'performance':
            response_parts.extend(self._generate_smart_performance_insights(query))
        elif intent['category'] == 'technology':
            response_parts.extend(self._generate_smart_technology_insights(query))
        elif intent['category'] == 'environmental':
            response_parts.extend(self._generate_smart_environmental_insights(query))
        elif intent['category'] == 'scm_practices':
            response_parts.extend(self._generate_smart_scm_insights(query))
        elif intent['category'] == 'comparative':
            response_parts.extend(self._generate_smart_comparative_insights(query))
        else:
            response_parts.extend(self._generate_smart_general_insights(query))
        
        # Add ML recommendations if available
        if 'ml_recommendations' in self.knowledge_base:
            response_parts.append("")
            response_parts.append("💡 **ML-Based Recommendations:**")
            response_parts.append(self._extract_relevant_recommendations(intent))
        
        return "\n".join(response_parts)
    
    def _generate_smart_performance_insights(self, query):
        """Generate smart performance insights based on actual data"""
        insights = []
        
        # Check if query mentions specific companies
        companies_mentioned = self._extract_company_names(query)
        
        if companies_mentioned:
            insights.append("🏆 **Company-Specific Performance Analysis:**")
            for company in companies_mentioned:
                company_data = self.df[self.df['company_name'].str.contains(company, case=False, na=False)]
                if not company_data.empty:
                    company_row = company_data.iloc[0]
                    insights.append(f"  📊 **{company_row['company_name']}:**")
                    insights.append(f"    • Operational Efficiency: {company_row['operational_efficiency_score']:.1f}")
                    insights.append(f"    • Environmental Impact: {company_row['environmental_impact_score']:.1f}")
                    insights.append(f"    • SCM Practice: {company_row['scm_practices']}")
                    insights.append(f"    • Technology: {company_row['technology_utilized']}")
                else:
                    insights.append(f"  ❌ Company '{company}' not found in dataset")
        else:
            # General performance insights
            insights.append("📈 **Overall Performance Insights:**")
            insights.append(f"  • Average Operational Score: {self.df['operational_efficiency_score'].mean():.1f}")
            insights.append(f"  • Top Score: {self.df['operational_efficiency_score'].max():.1f}")
            insights.append(f"  • Bottom Score: {self.df['operational_efficiency_score'].min():.1f}")
            
            # Top performers
            top_performers = self.df.nlargest(5, 'operational_efficiency_score')[['company_name', 'operational_efficiency_score']]
            insights.append("")
            insights.append("🏆 **Top 5 Performers:**")
            for i, (_, row) in enumerate(top_performers.iterrows(), 1):
                insights.append(f"  {i}. {row['company_name']}: {row['operational_efficiency_score']:.1f}")
        
        return insights
    
    def _generate_smart_technology_insights(self, query):
        """Generate smart technology insights based on actual data"""
        insights = []
        
        # Check if query mentions specific companies
        companies_mentioned = self._extract_company_names(query)
        
        if companies_mentioned:
            insights.append("🤖 **Company-Specific Technology Analysis:**")
            for company in companies_mentioned:
                company_data = self.df[self.df['company_name'].str.contains(company, case=False, na=False)]
                if not company_data.empty:
                    company_row = company_data.iloc[0]
                    insights.append(f"  📱 **{company_row['company_name']}:**")
                    insights.append(f"    • Technology Stack: {company_row['technology_utilized']}")
                    insights.append(f"    • AI Adoption: {'Yes' if 'AI' in str(company_row['technology_utilized']) else 'No'}")
                    insights.append(f"    • Blockchain: {'Yes' if 'Blockchain' in str(company_row['technology_utilized']) else 'No'}")
                    insights.append(f"    • ERP: {'Yes' if 'ERP' in str(company_row['technology_utilized']) else 'No'}")
        else:
            # General technology insights
            insights.append("🤖 **Technology Adoption Overview:**")
            
            # Count technology adoption
            ai_count = len(self.df[self.df['technology_utilized'].str.contains('AI', na=False)])
            blockchain_count = len(self.df[self.df['technology_utilized'].str.contains('Blockchain', na=False)])
            erp_count = len(self.df[self.df['technology_utilized'].str.contains('ERP', na=False)])
            robotics_count = len(self.df[self.df['technology_utilized'].str.contains('Robotics', na=False)])
            
            total_companies = len(self.df)
            
            insights.append(f"  • AI: {ai_count} companies ({(ai_count/total_companies)*100:.1f}%)")
            insights.append(f"  • Blockchain: {blockchain_count} companies ({(blockchain_count/total_companies)*100:.1f}%)")
            insights.append(f"  • ERP: {erp_count} companies ({(erp_count/total_companies)*100:.1f}%)")
            insights.append(f"  • Robotics: {robotics_count} companies ({(robotics_count/total_companies)*100:.1f}%)")
        
        return insights
    
    def _generate_smart_environmental_insights(self, query):
        """Generate smart environmental insights based on actual data"""
        insights = []
        
        # Check if query mentions specific companies
        companies_mentioned = self._extract_company_names(query)
        
        if companies_mentioned:
            insights.append("🌱 **Company-Specific Environmental Analysis:**")
            for company in companies_mentioned:
                company_data = self.df[self.df['company_name'].str.contains(company, case=False, na=False)]
                if not company_data.empty:
                    company_row = company_data.iloc[0]
                    insights.append(f"  🌿 **{company_row['company_name']}:**")
                    insights.append(f"    • Environmental Score: {company_row['environmental_impact_score']:.1f}")
                    insights.append(f"    • Sustainability Practices: {company_row['sustainability_practices']}")
                    insights.append(f"    • SCM Practice: {company_row['scm_practices']}")
                else:
                    insights.append(f"  ❌ Company '{company}' not found in dataset")
        else:
            # General environmental insights
            insights.append("🌱 **Environmental Performance Overview:**")
            insights.append(f"  • Average Environmental Score: {self.df['environmental_impact_score'].mean():.1f}")
            insights.append(f"  • Best Score: {self.df['environmental_impact_score'].max():.1f}")
            insights.append(f"  • Worst Score: {self.df['environmental_impact_score'].min():.1f}")
            
            # Top environmental performers
            top_env = self.df.nlargest(5, 'environmental_impact_score')[['company_name', 'environmental_impact_score']]
            insights.append("")
            insights.append("🌿 **Top 5 Environmental Performers:**")
            for i, (_, row) in enumerate(top_env.iterrows(), 1):
                insights.append(f"  {i}. {row['company_name']}: {row['environmental_impact_score']:.1f}")
        
        return insights
    
    def _generate_smart_scm_insights(self, query):
        """Generate smart SCM practice insights based on actual data"""
        insights = []
        
        # Check if query mentions specific companies
        companies_mentioned = self._extract_company_names(query)
        
        if companies_mentioned:
            insights.append("🔗 **Company-Specific SCM Analysis:**")
            for company in companies_mentioned:
                company_data = self.df[self.df['company_name'].str.contains(company, case=False, na=False)]
                if not company_data.empty:
                    company_row = company_data.iloc[0]
                    insights.append(f"  📋 **{company_row['company_name']}:**")
                    insights.append(f"    • SCM Practice: {company_row['scm_practices']}")
                    insights.append(f"    • Supply Chain Agility: {company_row['supply_chain_agility']}")
                    insights.append(f"    • Integration Level: {company_row['supply_chain_integration_level']}")
                    insights.append(f"    • Complexity Index: {company_row['supply_chain_complexity_index']}")
                else:
                    insights.append(f"  ❌ Company '{company}' not found in dataset")
        else:
            # General SCM insights
            insights.append("🔗 **SCM Practices Overview:**")
            
            # SCM practices distribution
            scm_dist = self.df['scm_practices'].value_counts()
            insights.append("  📊 **Practice Distribution:**")
            for practice, count in scm_dist.head(5).items():
                percentage = (count / len(self.df)) * 100
                insights.append(f"    • {practice}: {count} companies ({percentage:.1f}%)")
        
        return insights
    
    def _generate_smart_comparative_insights(self, query):
        """Generate smart comparative insights based on actual data"""
        insights = []
        
        # Check if query mentions specific companies
        companies_mentioned = self._extract_company_names(query)
        
        if len(companies_mentioned) >= 2:
            insights.append("🔍 **Company Comparison Analysis:**")
            
            company_data_list = []
            for company in companies_mentioned:
                company_data = self.df[self.df['company_name'].str.contains(company, case=False, na=False)]
                if not company_data.empty:
                    company_data_list.append(company_data.iloc[0])
            
            if len(company_data_list) >= 2:
                insights.append("  📊 **Performance Comparison:**")
                for company_row in company_data_list:
                    insights.append(f"    • **{company_row['company_name']}:**")
                    insights.append(f"      - Operational: {company_row['operational_efficiency_score']:.1f}")
                    insights.append(f"      - Environmental: {company_row['environmental_impact_score']:.1f}")
                    insights.append(f"      - SCM Practice: {company_row['scm_practices']}")
                    insights.append(f"      - Technology: {company_row['technology_utilized']}")
                
                # Calculate differences
                if len(company_data_list) == 2:
                    op_diff = company_data_list[0]['operational_efficiency_score'] - company_data_list[1]['operational_efficiency_score']
                    env_diff = company_data_list[0]['environmental_impact_score'] - company_data_list[1]['environmental_impact_score']
                    
                    insights.append("")
                    insights.append("  📈 **Performance Differences:**")
                    insights.append(f"    • Operational Efficiency: {op_diff:+.1f} points")
                    insights.append(f"    • Environmental Impact: {env_diff:+.1f} points")
            else:
                insights.append("  ❌ Could not find data for comparison")
        else:
            # General comparative insights
            insights.append("🔍 **General Comparative Insights:**")
            
            # AI vs Non-AI comparison
            ai_companies = self.df[self.df['technology_utilized'].str.contains('AI', na=False)]
            non_ai_companies = self.df[~self.df['technology_utilized'].str.contains('AI', na=False)]
            
            if len(ai_companies) > 0 and len(non_ai_companies) > 0:
                ai_avg = ai_companies['operational_efficiency_score'].mean()
                non_ai_avg = non_ai_companies['operational_efficiency_score'].mean()
                
                insights.append("  🤖 **AI vs Non-AI Performance:**")
                insights.append(f"    • AI Companies Average: {ai_avg:.1f}")
                insights.append(f"    • Non-AI Companies Average: {non_ai_avg:.1f}")
                insights.append(f"    • Difference: {ai_avg - non_ai_avg:+.1f} points")
        
        return insights
    
    def _generate_smart_general_insights(self, query):
        """Generate smart general insights based on actual data"""
        insights = []
        
        insights.append("📋 **Dataset Overview:**")
        insights.append(f"  • Total Companies: {len(self.df)}")
        insights.append(f"  • Data Dimensions: {len(self.df.columns)}")
        insights.append(f"  • Date Range: Latest data from 2025")
        
        # Check if query mentions specific companies
        companies_mentioned = self._extract_company_names(query)
        
        if companies_mentioned:
            insights.append("")
            insights.append("🏢 **Company Information:**")
            for company in companies_mentioned:
                company_data = self.df[self.df['company_name'].str.contains(company, case=False, na=False)]
                if not company_data.empty:
                    company_row = company_data.iloc[0]
                    insights.append(f"  📊 **{company_row['company_name']}:**")
                    insights.append(f"    • Industry: Supply Chain Management")
                    insights.append(f"    • Operational Score: {company_row['operational_efficiency_score']:.1f}")
                    insights.append(f"    • Environmental Score: {company_row['environmental_impact_score']:.1f}")
                    insights.append(f"    • Technology: {company_row['technology_utilized']}")
                else:
                    insights.append(f"  ❌ Company '{company}' not found in dataset")
        
        return insights
    
    def _extract_company_names(self, query):
        """Extract company names from query"""
        # Common company names in the dataset
        common_companies = [
            'Adobe', 'Apple', 'Microsoft', 'Google', 'Amazon', 'Facebook', 'Netflix',
            'Tesla', 'Intel', 'IBM', 'Oracle', 'Salesforce', 'Twitter', 'Alibaba',
            'Hulu', 'Looker', 'Abbott', 'Meditech', 'Bentley', 'Atlassian'
        ]
        
        companies_found = []
        query_lower = query.lower()
        
        for company in common_companies:
            if company.lower() in query_lower:
                companies_found.append(company)
        
        return companies_found
    
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
        
        print("\n🏢 **Company-Specific Questions:**")
        print("  • What is Adobe's operational efficiency score?")
        print("  • How does Apple perform environmentally?")
        print("  • Compare Microsoft and Google's SCM practices")
        print("  • What technology does Tesla use?")
        
        print("\n📊 **Performance Questions:**")
        print("  • Who are the top 5 performing companies?")
        print("  • What's the average environmental score?")
        print("  • Which companies use AI technology?")
        
        print("\n🔍 **Comparative Questions:**")
        print("  • Compare AI vs non-AI company performance")
        print("  • How do different SCM practices compare?")
        print("  • Which technology stack performs best?")
        
        print("\n💡 **TIPS:**")
        print("  • Ask about specific companies by name")
        print("  • Compare multiple companies")
        print("  • Ask for specific metrics or rankings")
        print()
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("🚀 **Smart SCM Green Logistics LLM Interface**")
        print("=" * 60)
        print("Ask me anything about your supply chain data!")
        print("I can answer specific questions about companies, compare performance, and more!")
        print("Type 'help' for sample queries, 'quit' to exit")
        print()
        
        self.show_sample_queries()
        
        while True:
            try:
                query = input("\n🤔 **Your Question:** ").strip()
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thanks for using the Smart SCM LLM Interface!")
                    break
                
                if query.lower() in ['help', 'samples', 'examples']:
                    self.show_sample_queries()
                    continue
                
                if not query:
                    print("❌ Please enter a question.")
                    continue
                
                # Analyze query and generate smart response
                intent = self.analyze_query_intent(query)
                response = self.generate_smart_response(query, intent)
                
                print("\n" + "=" * 70)
                print("🧠 **Smart LLM Response:**")
                print("=" * 70)
                print(response)
                print("=" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Smart SCM LLM Interface!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                print("Please try a different question.")

def main():
    """Main execution function"""
    smart_llm = SmartSCMLLMInterface()
    
    if smart_llm.df is not None:
        print("🎯 **Smart LLM Interface Ready!**")
        print("I can now answer specific questions about companies and provide real data insights!")
        print()
        
        # Run interactive mode
        smart_llm.interactive_mode()
    else:
        print("❌ Smart LLM Interface not available. Please check your data files.")

if __name__ == "__main__":
    main()
