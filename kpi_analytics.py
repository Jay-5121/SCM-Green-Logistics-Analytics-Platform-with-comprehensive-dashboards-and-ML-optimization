import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class SCMKPIAnalytics:
    def __init__(self):
        self.df = None
        self.output_dir = 'kpi_outputs'
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory for KPI results"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✅ Created output directory: {self.output_dir}")
    
    def load_data(self):
        """Load the cleaned SCM dataset"""
        print("🔄 Loading cleaned SCM dataset...")
        try:
            self.df = pd.read_csv('scm_cleaned.csv')
            print(f"✅ Dataset loaded: {self.df.shape}")
            return True
        except FileNotFoundError:
            print("❌ scm_cleaned.csv not found. Please run Phase 1 first.")
            return False
    
    def calculate_supply_chain_kpis(self):
        """Calculate comprehensive supply chain KPIs"""
        print("\n" + "="*60)
        print("📊 CALCULATING SUPPLY CHAIN KPIs")
        print("="*60)
        
        kpis = {}
        
        # 1. Cost per ton-km (proxy using operational efficiency and environmental score)
        if 'operational_efficiency_score' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            # Create a composite cost efficiency metric
            self.df['cost_efficiency_index'] = (self.df['operational_efficiency_score'] * 0.7 + 
                                               self.df['environmental_impact_score'] * 0.3) / 100
            kpis['cost_efficiency_index'] = {
                'mean': self.df['cost_efficiency_index'].mean(),
                'median': self.df['cost_efficiency_index'].median(),
                'std': self.df['cost_efficiency_index'].std()
            }
            print(f"✅ Cost Efficiency Index: {kpis['cost_efficiency_index']['mean']:.3f} ± {kpis['cost_efficiency_index']['std']:.3f}")
        
        # 2. Emissions per shipment (using environmental impact score as proxy)
        if 'environmental_impact_score' in self.df.columns:
            # Convert to emissions per shipment (lower score = higher emissions)
            self.df['emissions_per_shipment'] = (100 - self.df['environmental_impact_score']) / 10
            kpis['emissions_per_shipment'] = {
                'mean': self.df['emissions_per_shipment'].mean(),
                'median': self.df['emissions_per_shipment'].median(),
                'std': self.df['emissions_per_shipment'].std()
            }
            print(f"✅ Emissions per Shipment: {kpis['emissions_per_shipment']['mean']:.2f} ± {kpis['emissions_per_shipment']['std']:.2f}")
        
        # 3. Emissions per unit revenue (CO2e/USD proxy)
        if 'environmental_impact_score' in self.df.columns:
            # Create a sustainability efficiency metric
            self.df['sustainability_efficiency'] = self.df['environmental_impact_score'] / 100
            kpis['sustainability_efficiency'] = {
                'mean': self.df['sustainability_efficiency'].mean(),
                'median': self.df['sustainability_efficiency'].median(),
                'std': self.df['sustainability_efficiency'].std()
            }
            print(f"✅ Sustainability Efficiency: {kpis['sustainability_efficiency']['mean']:.3f} ± {kpis['sustainability_efficiency']['std']:.3f}")
        
        # 4. Supplier-level emission intensity
        if 'supplier_count' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            # Create supplier efficiency metric
            try:
                supplier_counts = pd.to_numeric(self.df['supplier_count'].astype(str).str.replace(',', ''), errors='coerce')
                self.df['supplier_emission_intensity'] = self.df['environmental_impact_score'] / supplier_counts.fillna(1)
                self.df['supplier_emission_intensity'] = self.df['supplier_emission_intensity'].fillna(self.df['supplier_emission_intensity'].median())
                
                kpis['supplier_emission_intensity'] = {
                    'mean': self.df['supplier_emission_intensity'].mean(),
                    'median': self.df['supplier_emission_intensity'].median(),
                    'std': self.df['supplier_emission_intensity'].std()
                }
                print(f"✅ Supplier Emission Intensity: {kpis['supplier_emission_intensity']['mean']:.3f} ± {kpis['supplier_emission_intensity']['std']:.3f}")
            except:
                print("⚠️ Could not calculate supplier emission intensity")
        
        # 5. Energy source dependency (using technology utilization as proxy)
        if 'technology_utilized' in self.df.columns:
            # Analyze technology adoption patterns
            tech_analysis = self.df['technology_utilized'].value_counts()
            kpis['technology_adoption'] = {
                'ai_adoption': len(self.df[self.df['technology_utilized'].str.contains('AI', na=False)]) / len(self.df),
                'blockchain_adoption': len(self.df[self.df['technology_utilized'].str.contains('Blockchain', na=False)]) / len(self.df),
                'robotics_adoption': len(self.df[self.df['technology_utilized'].str.contains('Robotics', na=False)]) / len(self.df)
            }
            print(f"✅ AI Adoption Rate: {kpis['technology_adoption']['ai_adoption']:.1%}")
            print(f"✅ Blockchain Adoption Rate: {kpis['technology_adoption']['blockchain_adoption']:.1%}")
            print(f"✅ Robotics Adoption Rate: {kpis['technology_adoption']['robotics_adoption']:.1%}")
        
        # 6. Time trend analysis (using lead time metrics as proxy)
        if 'lead_time_days' in self.df.columns:
            # Analyze lead time variability
            try:
                lead_times = pd.to_numeric(self.df['lead_time_days'], errors='coerce')
                valid_lead_times = lead_times.dropna()
                if len(valid_lead_times) > 0:
                    kpis['lead_time_analysis'] = {
                        'mean': valid_lead_times.mean(),
                        'median': valid_lead_times.median(),
                        'std': valid_lead_times.std(),
                        'efficiency': len(valid_lead_times[valid_lead_times <= 10]) / len(valid_lead_times)
                    }
                    print(f"✅ Lead Time Efficiency: {kpis['lead_time_analysis']['efficiency']:.1%}")
            except:
                print("⚠️ Could not analyze lead time metrics")
        
        return kpis
    
    def create_supplier_analysis(self):
        """Create comprehensive supplier analysis"""
        print("\n" + "="*60)
        print("🤝 SUPPLIER ANALYSIS")
        print("="*60)
        
        # Supplier collaboration analysis
        if 'supplier_collaboration_level' in self.df.columns:
            collab_analysis = self.df['supplier_collaboration_level'].value_counts()
            
            # Create bar plot
            plt.figure(figsize=(10, 6))
            collab_analysis.plot(kind='bar', color='lightcoral', edgecolor='black')
            plt.title('Supplier Collaboration Levels Across Companies', fontsize=14, fontweight='bold')
            plt.xlabel('Collaboration Level', fontsize=12)
            plt.ylabel('Number of Companies', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/supplier_collaboration_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Supplier collaboration analysis plot saved")
            
            # Print summary
            print("\n📊 Supplier Collaboration Summary:")
            for level, count in collab_analysis.items():
                percentage = (count / len(self.df)) * 100
                print(f"  {level}: {count} companies ({percentage:.1f}%)")
    
    def create_route_efficiency_analysis(self):
        """Analyze route efficiency and transportation metrics"""
        print("\n" + "="*60)
        print("🚚 ROUTE EFFICIENCY ANALYSIS")
        print("="*60)
        
        # Transportation cost efficiency analysis
        if 'transportation_cost_efficiency_' in self.df.columns:
            transport_analysis = self.df['transportation_cost_efficiency_'].value_counts().sort_index()
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(self.df['transportation_cost_efficiency_'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
            plt.title('Distribution of Transportation Cost Efficiency', fontsize=14, fontweight='bold')
            plt.xlabel('Efficiency Score', fontsize=12)
            plt.ylabel('Number of Companies', fontsize=12)
            plt.axvline(self.df['transportation_cost_efficiency_'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {self.df["transportation_cost_efficiency_"].mean():.1f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/transportation_efficiency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Transportation efficiency analysis plot saved")
            
            # Print summary
            print(f"\n📊 Transportation Cost Efficiency:")
            print(f"  Mean: {self.df['transportation_cost_efficiency_'].mean():.1f}")
            print(f"  High performers (>85): {len(self.df[self.df['transportation_cost_efficiency_'] > 85])}")
            print(f"  Low performers (<80): {len(self.df[self.df['transportation_cost_efficiency_'] < 80])}")
    
    def create_time_series_analysis(self):
        """Create time series analysis of logistics costs vs emissions"""
        print("\n" + "="*60)
        print("⏰ TIME SERIES ANALYSIS")
        print("="*60)
        
        # Create scatter plot of operational efficiency vs environmental impact
        if 'operational_efficiency_score' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            plt.figure(figsize=(12, 8))
            
            # Color code by SCM practice
            if 'scm_practices' in self.df.columns:
                practices = self.df['scm_practices'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(practices)))
                
                for i, practice in enumerate(practices):
                    subset = self.df[self.df['scm_practices'] == practice]
                    plt.scatter(subset['operational_efficiency_score'], subset['environmental_impact_score'],
                              c=[colors[i]], label=practice, alpha=0.7, s=50)
            else:
                plt.scatter(self.df['operational_efficiency_score'], self.df['environmental_impact_score'],
                          alpha=0.7, s=50, color='skyblue')
            
            plt.xlabel('Operational Efficiency Score', fontsize=12)
            plt.ylabel('Environmental Impact Score', fontsize=12)
            plt.title('Operational Efficiency vs Environmental Impact', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/efficiency_vs_environmental_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Efficiency vs Environmental impact scatter plot saved")
            
            # Calculate correlation
            correlation = self.df['operational_efficiency_score'].corr(self.df['environmental_impact_score'])
            print(f"\n📊 Correlation between Efficiency and Environmental Impact: {correlation:.3f}")
    
    def create_kpi_dashboard(self):
        """Create comprehensive KPI dashboard"""
        print("\n" + "="*60)
        print("📊 CREATING KPI DASHBOARD")
        print("="*60)
        
        # Create summary tables
        summary_data = []
        
        # Company performance summary
        if 'company_name' in self.df.columns and 'operational_efficiency_score' in self.df.columns:
            top_performers = self.df.nlargest(20, 'operational_efficiency_score')[['company_name', 'operational_efficiency_score', 'environmental_impact_score']]
            summary_data.append(('Top 20 Operational Performers', top_performers))
        
        # SCM practices summary
        if 'scm_practices' in self.df.columns:
            practices_summary = self.df['scm_practices'].value_counts().reset_index()
            practices_summary.columns = ['SCM Practice', 'Company Count']
            practices_summary['Percentage'] = (practices_summary['Company Count'] / len(self.df) * 100).round(1)
            summary_data.append(('SCM Practices Distribution', practices_summary))
        
        # Environmental performance summary
        if 'environmental_impact_score' in self.df.columns:
            env_summary = self.df.groupby('scm_practices')['environmental_impact_score'].agg(['mean', 'std', 'count']).round(2)
            env_summary.columns = ['Average Score', 'Std Dev', 'Company Count']
            summary_data.append(('Environmental Performance by SCM Practice', env_summary))
        
        # Save summary tables to Excel
        with pd.ExcelWriter(f'{self.output_dir}/scm_kpis.xlsx', engine='openpyxl') as writer:
            # Executive Summary
            exec_summary = pd.DataFrame({
                'Metric': ['Total Companies', 'Average Operational Efficiency', 'Average Environmental Score', 'Top SCM Practice'],
                'Value': [
                    len(self.df),
                    f"{self.df['operational_efficiency_score'].mean():.1f}",
                    f"{self.df['environmental_impact_score'].mean():.1f}",
                    self.df['scm_practices'].mode().iloc[0] if 'scm_practices' in self.df.columns else 'N/A'
                ]
            })
            exec_summary.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # KPI Dashboard
            kpi_dashboard = pd.DataFrame({
                'KPI Category': ['Cost Efficiency', 'Environmental Impact', 'Operational Performance', 'Technology Adoption'],
                'Key Metric': ['Cost Efficiency Index', 'Emissions per Shipment', 'Operational Efficiency Score', 'AI Adoption Rate'],
                'Current Value': [
                    f"{self.df['cost_efficiency_index'].mean():.3f}" if 'cost_efficiency_index' in self.df.columns else 'N/A',
                    f"{self.df['emissions_per_shipment'].mean():.2f}" if 'emissions_per_shipment' in self.df.columns else 'N/A',
                    f"{self.df['operational_efficiency_score'].mean():.1f}",
                    f"{len(self.df[self.df['technology_utilized'].str.contains('AI', na=False)]) / len(self.df):.1%}" if 'technology_utilized' in self.df.columns else 'N/A'
                ],
                'Target': ['>0.75', '<2.5', '>85', '>80%'],
                'Status': ['On Track' if 'cost_efficiency_index' in self.df.columns and self.df['cost_efficiency_index'].mean() > 0.75 else 'Needs Improvement',
                          'On Track' if 'emissions_per_shipment' in self.df.columns and self.df['emissions_per_shipment'].mean() < 2.5 else 'Needs Improvement',
                          'On Track' if self.df['operational_efficiency_score'].mean() > 85 else 'Needs Improvement',
                          'On Track' if 'technology_utilized' in self.df.columns and len(self.df[self.df['technology_utilized'].str.contains('AI', na=False)]) / len(self.df) > 0.8 else 'Needs Improvement']
            })
            kpi_dashboard.to_excel(writer, sheet_name='KPI Dashboard', index=False)
            
            # Save other summary tables
            for name, data in summary_data:
                sheet_name = name[:31] if len(name) > 31 else name  # Excel sheet name limit
                data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print("✅ KPI dashboard saved as 'scm_kpis.xlsx'")
        
        # Create KPI summary visualization
        self.create_kpi_visualizations()
    
    def create_kpi_visualizations(self):
        """Create KPI summary visualizations"""
        print("\n" + "="*60)
        print("🎨 CREATING KPI VISUALIZATIONS")
        print("="*60)
        
        # Create subplot with multiple KPI charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SCM Green Logistics - KPI Dashboard Overview', fontsize=16, fontweight='bold')
        
        # 1. Operational Efficiency Distribution
        if 'operational_efficiency_score' in self.df.columns:
            axes[0, 0].hist(self.df['operational_efficiency_score'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            axes[0, 0].set_title('Operational Efficiency Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('Efficiency Score')
            axes[0, 0].set_ylabel('Number of Companies')
            axes[0, 0].axvline(self.df['operational_efficiency_score'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {self.df["operational_efficiency_score"].mean():.1f}')
            axes[0, 0].legend()
        
        # 2. Environmental Impact by SCM Practice
        if 'scm_practices' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            practice_env = self.df.groupby('scm_practices')['environmental_impact_score'].mean().sort_values(ascending=False)
            practice_env.plot(kind='bar', ax=axes[0, 1], color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Environmental Impact by SCM Practice', fontweight='bold')
            axes[0, 1].set_xlabel('SCM Practice')
            axes[0, 1].set_ylabel('Average Environmental Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Technology Adoption
        if 'technology_utilized' in self.df.columns:
            tech_counts = {}
            for tech in ['AI', 'Blockchain', 'Robotics', 'ERP']:
                tech_counts[tech] = len(self.df[self.df['technology_utilized'].str.contains(tech, na=False)])
            
            tech_df = pd.Series(tech_counts)
            tech_df.plot(kind='bar', ax=axes[1, 0], color='gold', edgecolor='black')
            axes[1, 0].set_title('Technology Adoption Rates', fontweight='bold')
            axes[1, 0].set_xlabel('Technology')
            axes[1, 0].set_ylabel('Number of Companies')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Cost vs Environmental Efficiency
        if 'cost_efficiency_index' in self.df.columns and 'environmental_impact_score' in self.df.columns:
            axes[1, 1].scatter(self.df['cost_efficiency_index'], self.df['environmental_impact_score'], alpha=0.6, s=30)
            axes[1, 1].set_title('Cost Efficiency vs Environmental Impact', fontweight='bold')
            axes[1, 1].set_xlabel('Cost Efficiency Index')
            axes[1, 1].set_ylabel('Environmental Impact Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/kpi_dashboard_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ KPI dashboard overview visualization saved")
    
    def generate_kpi_report(self):
        """Generate comprehensive KPI report"""
        print("\n" + "="*60)
        print("📋 GENERATING KPI REPORT")
        print("="*60)
        
        report = []
        report.append("=" * 80)
        report.append("SCM GREEN LOGISTICS - KPI ANALYTICS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("📊 EXECUTIVE SUMMARY:")
        report.append(f"  Total Companies Analyzed: {len(self.df)}")
        report.append(f"  Average Operational Efficiency: {self.df['operational_efficiency_score'].mean():.1f}")
        report.append(f"  Average Environmental Score: {self.df['environmental_impact_score'].mean():.1f}")
        report.append("")
        
        # Key Performance Indicators
        report.append("🎯 KEY PERFORMANCE INDICATORS:")
        
        if 'cost_efficiency_index' in self.df.columns:
            report.append(f"  • Cost Efficiency Index: {self.df['cost_efficiency_index'].mean():.3f}")
        
        if 'emissions_per_shipment' in self.df.columns:
            report.append(f"  • Emissions per Shipment: {self.df['emissions_per_shipment'].mean():.2f}")
        
        if 'sustainability_efficiency' in self.df.columns:
            report.append(f"  • Sustainability Efficiency: {self.df['sustainability_efficiency'].mean():.3f}")
        
        report.append("")
        
        # Top Performers
        report.append("🏆 TOP PERFORMERS:")
        if 'operational_efficiency_score' in self.df.columns:
            top_3 = self.df.nlargest(3, 'operational_efficiency_score')[['company_name', 'operational_efficiency_score']]
            for idx, row in top_3.iterrows():
                report.append(f"  • {row['company_name']}: {row['operational_efficiency_score']:.1f}")
        
        report.append("")
        
        # Recommendations
        report.append("💡 STRATEGIC RECOMMENDATIONS:")
        report.append("  1. Benchmark against top performers in operational efficiency")
        report.append("  2. Focus on SCM practices that show highest environmental scores")
        report.append("  3. Invest in AI and blockchain technologies for supply chain optimization")
        report.append("  4. Develop supplier collaboration programs to improve emission intensity")
        report.append("  5. Implement cost-environmental efficiency tracking systems")
        
        # Save report
        with open(f'{self.output_dir}/kpi_analytics_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("✅ KPI analytics report saved")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("KPI ANALYTICS REPORT SUMMARY")
        print("=" * 80)
        for line in report:
            print(line)
    
    def run_full_kpi_analysis(self):
        """Run the complete KPI analytics pipeline"""
        print("🚀 Starting SCM Green Logistics KPI Analytics")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Run all analyses
        kpis = self.calculate_supply_chain_kpis()
        self.create_supplier_analysis()
        self.create_route_efficiency_analysis()
        self.create_time_series_analysis()
        self.create_kpi_dashboard()
        self.generate_kpi_report()
        
        print("\n" + "=" * 70)
        print("✅ KPI ANALYTICS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 All outputs saved in: {self.output_dir}/")
        print("📊 Excel dashboard: scm_kpis.xlsx")
        print("🎯 Ready for Phase 4: Machine Learning / Optimization")
        
        return True

def main():
    """Main execution function"""
    kpi_analytics = SCMKPIAnalytics()
    success = kpi_analytics.run_full_kpi_analysis()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Review KPI outputs in kpi_outputs/ folder")
        print("2. Examine Excel dashboard: scm_kpis.xlsx")
        print("3. Proceed to Phase 4: Machine Learning / Optimization")
    else:
        print("\n❌ KPI analytics failed. Please check the errors above.")

if __name__ == "__main__":
    main()
