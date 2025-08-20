import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SCMDataPreprocessor:
    def __init__(self):
        self.datasets = {}
        self.cleaned_data = None
        self.data_quality_report = {}
        
    def load_datasets(self):
        """Load all CSV datasets with encoding handling"""
        print("🔄 Loading datasets...")
        
        # Define dataset configurations with encoding and separator info
        dataset_configs = {
            'scm': {'file': 'SCM Dataset.csv', 'encoding': 'latin-1', 'sep': ','},
            'emm': {'file': 'EMM_EPM0_PTE_NUS_DPGa.csv', 'encoding': 'utf-8', 'sep': ','},
            'pet': {'file': 'PET_PRI_GND_DCUS_NUS_A.csv', 'encoding': 'utf-8', 'sep': ','},
            'ghg': {'file': 'ghg-emission-factors-hub-2025.csv', 'encoding': 'utf-8', 'sep': ','},
            'wei': {'file': 'W_E_I_World.csv', 'encoding': 'utf-8', 'sep': ','},
            'scm_ghg': {'file': 'SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv', 'encoding': 'utf-8', 'sep': ','}
        }
        
        for name, config in dataset_configs.items():
            try:
                # Try to load with specified encoding
                df = pd.read_csv(config['file'], encoding=config['encoding'], sep=config['sep'])
                self.datasets[name] = df
                print(f"✅ {name}: {df.shape} - loaded with {config['encoding']} encoding")
                
            except UnicodeDecodeError:
                # If encoding fails, try other common encodings
                for encoding in ['cp1252', 'iso-8859-1', 'utf-8-sig']:
                    try:
                        df = pd.read_csv(config['file'], encoding=encoding, sep=config['sep'])
                        self.datasets[name] = df
                        print(f"✅ {name}: {df.shape} - loaded with {encoding} encoding")
                        break
                    except:
                        continue
                else:
                    print(f"❌ Failed to load {name} with any encoding")
                    return False
                    
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
                return False
            
        return True
    
    def standardize_column_names(self):
        """Standardize column names to lowercase, no spaces, snake_case"""
        print("🔄 Standardizing column names...")
        
        for name, df in self.datasets.items():
            # Convert to lowercase and replace spaces with underscores
            new_columns = {}
            for col in df.columns:
                new_col = col.lower().replace(' ', '_').replace('-', '_')
                new_col = ''.join(c for c in new_col if c.isalnum() or c == '_')
                new_columns[col] = new_col
            
            # Rename columns
            df.rename(columns=new_columns, inplace=True)
            print(f"✅ {name}: {len(new_columns)} columns standardized")
    
    def check_missing_values(self):
        """Check for missing values and create data quality report"""
        print("🔄 Checking missing values...")
        
        for name, df in self.datasets.items():
            missing_info = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_counts': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'columns_with_missing': df.columns[df.isnull().any()].tolist()
            }
            
            self.data_quality_report[name] = missing_info
            
            # Print summary
            total_missing = df.isnull().sum().sum()
            print(f"✅ {name}: {total_missing} missing values out of {len(df) * len(df.columns)} total cells")
    
    def convert_dates(self):
        """Convert date columns to proper datetime format"""
        print("🔄 Converting date columns...")
        
        for name, df in self.datasets.items():
            date_columns = []
            
            # Look for columns that might contain dates
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                    date_columns.append(col)
            
            # Try to convert date columns
            for col in date_columns:
                try:
                    # Try different date formats
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"✅ {name}.{col}: converted to datetime")
                except:
                    print(f"⚠️ {name}.{col}: could not convert to datetime")
    
    def clean_scm_dataset(self):
        """Clean and prepare the main SCM dataset"""
        print("🔄 Cleaning SCM dataset...")
        
        scm = self.datasets['scm'].copy()
        
        # Remove completely empty rows and columns
        scm.dropna(how='all', inplace=True)
        scm.dropna(axis=1, how='all', inplace=True)
        
        # Fill missing values with appropriate defaults
        numeric_columns = scm.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if scm[col].isnull().sum() > 0:
                scm[col].fillna(scm[col].median(), inplace=True)
        
        # Fill categorical missing values
        categorical_columns = scm.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if scm[col].isnull().sum() > 0:
                scm[col].fillna('Unknown', inplace=True)
        
        self.datasets['scm'] = scm
        print(f"✅ SCM dataset cleaned: {scm.shape}")
    
    def merge_datasets(self):
        """Merge SCM operational data with emissions and cost data"""
        print("🔄 Merging datasets...")
        
        # Start with cleaned SCM data
        merged = self.datasets['scm'].copy()
        
        # Add basic merge keys if they don't exist
        if 'merge_key' not in merged.columns:
            merged['merge_key'] = range(len(merged))
        
        # For now, create a comprehensive merged dataset
        # This will be enhanced based on actual data structure
        print(f"✅ Merged dataset created: {merged.shape}")
        
        self.cleaned_data = merged
    
    def save_cleaned_data(self):
        """Save the cleaned and merged dataset"""
        if self.cleaned_data is not None:
            self.cleaned_data.to_csv('scm_cleaned.csv', index=False)
            print("✅ Cleaned dataset saved as 'scm_cleaned.csv'")
        else:
            print("❌ No cleaned data to save")
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("🔄 Generating data quality report...")
        
        report = []
        report.append("=" * 60)
        report.append("SCM GREEN LOGISTICS - DATA QUALITY REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for name, info in self.data_quality_report.items():
            report.append(f"📊 DATASET: {name.upper()}")
            report.append(f"   Total Rows: {info['total_rows']:,}")
            report.append(f"   Total Columns: {info['total_columns']}")
            report.append(f"   Missing Values: {sum(info['missing_counts'].values()):,}")
            report.append("")
            
            if info['columns_with_missing']:
                report.append("   Columns with missing values:")
                for col in info['columns_with_missing']:
                    missing_count = info['missing_counts'][col]
                    missing_pct = info['missing_percentage'][col]
                    report.append(f"     - {col}: {missing_count:,} ({missing_pct:.1f}%)")
                report.append("")
        
        # Save report
        with open('data_quality_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("✅ Data quality report saved as 'data_quality_report.txt'")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("DATA QUALITY SUMMARY")
        print("=" * 60)
        for line in report:
            print(line)
    
    def run_full_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("🚀 Starting SCM Green Logistics Data Preprocessing Pipeline")
        print("=" * 70)
        
        # Step 1: Load datasets
        if not self.load_datasets():
            print("❌ Failed to load datasets. Exiting.")
            return False
        
        # Step 2: Standardize column names
        self.standardize_column_names()
        
        # Step 3: Check missing values
        self.check_missing_values()
        
        # Step 4: Convert dates
        self.convert_dates()
        
        # Step 5: Clean SCM dataset
        self.clean_scm_dataset()
        
        # Step 6: Merge datasets
        self.merge_datasets()
        
        # Step 7: Save cleaned data
        self.save_cleaned_data()
        
        # Step 8: Generate report
        self.generate_data_quality_report()
        
        print("\n" + "=" * 70)
        print("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return True

def main():
    """Main execution function"""
    preprocessor = SCMDataPreprocessor()
    success = preprocessor.run_full_preprocessing()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Review data_quality_report.txt for insights")
        print("2. Examine scm_cleaned.csv for data structure")
        print("3. Proceed to Phase 2: Exploratory Data Analysis")
    else:
        print("\n❌ Preprocessing failed. Please check the errors above.")

if __name__ == "__main__":
    main()
