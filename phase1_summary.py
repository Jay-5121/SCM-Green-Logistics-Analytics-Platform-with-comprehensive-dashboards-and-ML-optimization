import pandas as pd
import os

def show_phase1_results():
    """Show summary of Phase 1 results"""
    print("=" * 70)
    print("🎯 PHASE 1 COMPLETED: DATA PREPROCESSING RESULTS")
    print("=" * 70)
    
    # Check what files we have
    if os.path.exists('scm_cleaned.csv'):
        df = pd.read_csv('scm_cleaned.csv')
        print(f"✅ CLEANED DATASET CREATED: scm_cleaned.csv")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Sample columns: {list(df.columns[:5])}...")
        print()
        
        # Show data types
        print("📊 DATA TYPES:")
        for col, dtype in df.dtypes.items():
            print(f"   {col}: {dtype}")
        print()
        
        # Show sample data
        print("📋 SAMPLE DATA (first 3 rows):")
        print(df.head(3).to_string())
        print()
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("⚠️ MISSING VALUES:")
            for col, count in missing[missing > 0].items():
                print(f"   {col}: {count} missing")
        else:
            print("✅ NO MISSING VALUES FOUND")
        print()
        
    else:
        print("❌ Cleaned dataset not found")
    
    # Check other outputs
    if os.path.exists('data_quality_report.txt'):
        print("✅ Data quality report generated")
    else:
        print("⚠️ Data quality report not generated")
    
    print("\n" + "=" * 70)
    print("🚀 READY FOR PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

if __name__ == "__main__":
    show_phase1_results()
