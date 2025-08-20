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

class SCMExploratoryAnalysis:
    def __init__(self):
        self.df = None
        self.output_dir = 'eda_outputs'
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory for EDA results"""
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
    
    def basic_data_overview(self):
        """Provide basic overview of the dataset"""
        print("\n" + "="*60)
        print("📊 BASIC DATA OVERVIEW")
        print("="*60)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total Companies: {len(self.df)}")
        print(f"Total Features: {len(self.df.columns)}")
        
        print("\n📋 Column Information:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            print(f"  {col}: {dtype} ({unique_count} unique values)")
    
    def analyze_scm_practices(self):
        """Analyze SCM practices distribution"""
        print("\n" + "="*60)
        print("🏭 SCM PRACTICES ANALYSIS")
        print("="*60)
        
        if 'scm_practices' in self.df.columns:
            practices = self.df['scm_practices'].value_counts()
            
            # Create bar plot
            plt.figure(figsize=(12, 6))
            practices.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title('Distribution of SCM Practices Across Companies', fontsize=14, fontweight='bold')
            plt.xlabel('SCM Practice Type', fontsize=12)
            plt.ylabel('Number of Companies', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/scm_practices_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ SCM practices distribution plot saved")
            print("\nSCM Practices Summary:")
            for practice, count in practices.items():
                percentage = (count / len(self.df)) * 100
                print(f"  {practice}: {count} companies ({percentage:.1f}%)")
    
    def analyze_environmental_metrics(self):
        """Analyze environmental impact and sustainability metrics"""
        print("\n" + "="*60)
        print("🌱 ENVIRONMENTAL IMPACT ANALYSIS")
        print("="*60)
        
        env_cols = ['environmental_impact_score', 'sustainability_practices']
        
        for col in env_cols:
            if col in self.df.columns:
                if col == 'environmental_impact_score':
                    # Numeric analysis
                    print(f"\n📊 {col.replace('_', ' ').title()}:")
                    print(f"  Mean: {self.df[col].mean():.2f}")
                    print(f"  Median: {self.df[col].median():.2f}")
                    print(f"  Std Dev: {self.df[col].std():.2f}")
                    print(f"  Min: {self.df[col].min():.2f}")
                    print(f"  Max: {self.df[col].max():.2f}")
                    
                    # Create histogram
                    plt.figure(figsize=(10, 6))
                    plt.hist(self.df[col], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
                    plt.title(f'Distribution of {col.replace("_", " ").title()}', fontsize=14, fontweight='bold')
                    plt.xlabel('Score', fontsize=12)
                    plt.ylabel('Number of Companies', fontsize=12)
                    plt.axvline(self.df[col].mean(), color='red', linestyle='--', label=f'Mean: {self.df[col].mean():.2f}')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'{self.output_dir}/{col}_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                else:
                    # Categorical analysis
                    practices = self.df[col].value_counts()
                    print(f"\n📊 {col.replace('_', ' ').title()}:")
                    for practice, count in practices.items():
                        percentage = (count / len(self.df)) * 100
                        print(f"  {practice}: {count} companies ({percentage:.1f}%)")
                    
                    # Create pie chart
                    plt.figure(figsize=(10, 8))
                    plt.pie(practices.values, labels=practices.index, autopct='%1.1f%%', startangle=90)
                    plt.title(f'Distribution of {col.replace("_", " ").title()}', fontsize=14, fontweight='bold')
                    plt.axis('equal')
                    plt.tight_layout()
                    plt.savefig(f'{self.output_dir}/{col}_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
                
                print(f"✅ {col} analysis plot saved")
    
    def correlation_analysis(self):
        """Perform correlation analysis between key metrics"""
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Key Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Correlation matrix saved")
        
        # Find top correlations
        print("\n🔍 Top Correlations (|r| > 0.3):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    print(f"  {col1} ↔ {col2}: {corr_value:.3f}")
    
    def top_performers_analysis(self):
        """Identify top performing companies in different categories"""
        print("\n" + "="*60)
        print("🏆 TOP PERFORMERS ANALYSIS")
        print("="*60)
        
        # Top environmental performers
        if 'environmental_impact_score' in self.df.columns:
            top_env = self.df.nlargest(10, 'environmental_impact_score')[['company_name', 'environmental_impact_score']]
            print("\n🌱 Top 10 Environmental Performers:")
            for idx, row in top_env.iterrows():
                print(f"  {row['company_name']}: {row['environmental_impact_score']:.1f}")
        
        # Top operational efficiency
        if 'operational_efficiency_score' in self.df.columns:
            top_eff = self.df.nlargest(10, 'operational_efficiency_score')[['company_name', 'operational_efficiency_score']]
            print("\n⚡ Top 10 Operational Efficiency:")
            for idx, row in top_eff.iterrows():
                print(f"  {row['company_name']}: {row['operational_efficiency_score']:.1f}")
    
    def generate_eda_summary(self):
        """Generate comprehensive EDA summary"""
        print("\n" + "="*60)
        print("📋 GENERATING EDA SUMMARY")
        print("="*60)
        
        summary = []
        summary.append("=" * 80)
        summary.append("SCM GREEN LOGISTICS - EXPLORATORY DATA ANALYSIS SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        summary.append("📊 DATASET OVERVIEW:")
        summary.append(f"  Total Companies: {len(self.df)}")
        summary.append(f"  Total Features: {len(self.df.columns)}")
        summary.append("")
        
        # Key insights
        if 'environmental_impact_score' in self.df.columns:
            env_mean = self.df['environmental_impact_score'].mean()
            env_std = self.df['environmental_impact_score'].std()
            summary.append("🌱 ENVIRONMENTAL INSIGHTS:")
            summary.append(f"  Average Environmental Score: {env_mean:.2f} ± {env_std:.2f}")
            summary.append(f"  Companies above average: {len(self.df[self.df['environmental_impact_score'] > env_mean])}")
            summary.append("")
        
        if 'operational_efficiency_score' in self.df.columns:
            eff_mean = self.df['operational_efficiency_score'].mean()
            summary.append("⚡ OPERATIONAL INSIGHTS:")
            summary.append(f"  Average Operational Efficiency: {eff_mean:.2f}")
            summary.append(f"  High performers (>80): {len(self.df[self.df['operational_efficiency_score'] > 80])}")
            summary.append("")
        
        # SCM practices distribution
        if 'scm_practices' in self.df.columns:
            practices = self.df['scm_practices'].value_counts()
            summary.append("🏭 SCM PRACTICES DISTRIBUTION:")
            for practice, count in practices.items():
                percentage = (count / len(self.df)) * 100
                summary.append(f"  {practice}: {count} companies ({percentage:.1f}%)")
            summary.append("")
        
        summary.append("📁 OUTPUT FILES GENERATED:")
        summary.append(f"  All plots saved in: {self.output_dir}/")
        summary.append("  - SCM practices distribution")
        summary.append("  - Environmental impact analysis")
        summary.append("  - Correlation matrix")
        summary.append("")
        
        summary.append("🎯 KEY RECOMMENDATIONS:")
        summary.append("  1. Focus on companies with high environmental scores for best practices")
        summary.append("  2. Analyze correlation between environmental and operational performance")
        summary.append("  3. Identify supply chain practices that drive sustainability")
        summary.append("  4. Benchmark against top performers in each category")
        
        # Save summary
        with open(f'{self.output_dir}/eda_summary.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
        
        print("✅ EDA summary saved")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("EDA SUMMARY")
        print("=" * 80)
        for line in summary:
            print(line)
    
    def run_full_eda(self):
        """Run the complete EDA pipeline"""
        print("🚀 Starting SCM Green Logistics Exploratory Data Analysis")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Run all analyses
        self.basic_data_overview()
        self.analyze_scm_practices()
        self.analyze_environmental_metrics()
        self.correlation_analysis()
        self.top_performers_analysis()
        self.generate_eda_summary()
        
        print("\n" + "=" * 70)
        print("✅ EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 All outputs saved in: {self.output_dir}/")
        print("🎯 Ready for Phase 3: Analytics & Managerial KPIs")
        
        return True

def main():
    """Main execution function"""
    eda = SCMExploratoryAnalysis()
    success = eda.run_full_eda()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Review EDA outputs in eda_outputs/ folder")
        print("2. Examine correlation insights")
        print("3. Proceed to Phase 3: Analytics & Managerial KPIs")
    else:
        print("\n❌ EDA failed. Please check the errors above.")

if __name__ == "__main__":
    main()
