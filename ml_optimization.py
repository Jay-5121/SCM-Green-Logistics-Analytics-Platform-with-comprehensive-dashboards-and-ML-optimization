import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class SCMMLOptimization:
    def __init__(self):
        self.df = None
        self.output_dir = 'ml_outputs'
        self.models = {}
        self.results = {}
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory for ML results"""
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
    
    def prepare_features(self):
        """Prepare features for machine learning models"""
        print("\n" + "="*60)
        print("🔧 PREPARING FEATURES FOR ML MODELS")
        print("(Multi-objective framing: optimizing both cost efficiency and environmental performance)")
        print("="*60)
        
        # Create feature engineering
        df_ml = self.df.copy()
        
        # 1. Encode categorical variables
        categorical_cols = ['scm_practices', 'supply_chain_agility', 'supply_chain_integration_level', 
                           'sustainability_practices', 'supply_chain_complexity_index', 'technology_utilized']
        
        label_encoders = {}
        for col in categorical_cols:
            if col in df_ml.columns:
                le = LabelEncoder()
                df_ml[f'{col}_encoded'] = le.fit_transform(df_ml[col].astype(str))
                label_encoders[col] = le
                print(f"✅ Encoded {col}: {len(le.classes_)} categories")
        
        # 2. Handle supplier count (convert to numeric)
        if 'supplier_count' in df_ml.columns:
            try:
                supplier_counts = pd.to_numeric(df_ml['supplier_count'].astype(str).str.replace(',', ''), errors='coerce')
                df_ml['supplier_count_numeric'] = supplier_counts.fillna(supplier_counts.median())
                print("✅ Converted supplier_count to numeric")
            except:
                print("⚠️ Could not convert supplier_count")
        
        # 3. Create composite features
        if 'operational_efficiency_score' in df_ml.columns and 'environmental_impact_score' in df_ml.columns:
            df_ml['sustainability_efficiency'] = df_ml['environmental_impact_score'] / 100
            df_ml['cost_efficiency_index'] = (df_ml['operational_efficiency_score'] * 0.7 + 
                                             df_ml['environmental_impact_score'] * 0.3) / 100
            print("✅ Created composite efficiency features (cost vs carbon components)")
        
        # 4. Create technology adoption features
        if 'technology_utilized' in df_ml.columns:
            tech_features = ['AI', 'Blockchain', 'Robotics', 'ERP']
            for tech in tech_features:
                df_ml[f'has_{tech.lower()}'] = df_ml['technology_utilized'].str.contains(tech, na=False).astype(int)
            print("✅ Created technology adoption features")
        
        # 5. Select final features for ML
        feature_cols = []
        target_cols = []
        
        # Features
        numeric_features = ['inventory_turnover_ratio', 'order_fulfillment_rate_', 'customer_satisfaction_',
                          'environmental_impact_score', 'inventory_accuracy_', 'transportation_cost_efficiency_',
                          'operational_efficiency_score', 'revenue_growth_rate_out_of_15', 'supply_chain_risk_',
                          'supply_chain_resilience_score', 'supplier_relationship_score']
        
        for col in numeric_features:
            if col in df_ml.columns:
                feature_cols.append(col)
        
        # Add encoded categorical features
        for col in categorical_cols:
            if f'{col}_encoded' in df_ml.columns:
                feature_cols.append(f'{col}_encoded')
        
        # Add technology features
        for tech in ['ai', 'blockchain', 'robotics', 'erp']:
            if f'has_{tech}' in df_ml.columns:
                feature_cols.append(f'has_{tech}')
        
        # Add composite features
        if 'sustainability_efficiency' in df_ml.columns:
            feature_cols.append('sustainability_efficiency')
        if 'cost_efficiency_index' in df_ml.columns:
            feature_cols.append('cost_efficiency_index')
        
        print(f"\n📊 Feature Engineering Summary:")
        print(f"  Total Features: {len(feature_cols)}")
        print(f"  Targets: operational_efficiency_score, environmental_impact_score, cost_efficiency_index")
        
        # Create feature matrix and targets
        X = df_ml[feature_cols].fillna(0)  # Fill NaN with 0 for ML
        y_operational = df_ml['operational_efficiency_score'] if 'operational_efficiency_score' in df_ml.columns else None
        y_environmental = df_ml['environmental_impact_score'] if 'environmental_impact_score' in df_ml.columns else None
        y_cost = df_ml['cost_efficiency_index'] if 'cost_efficiency_index' in df_ml.columns else None
        
        return X, y_operational, y_environmental, y_cost, feature_cols
    
    def build_ml_models(self, X, y_operational, y_environmental, y_cost, feature_cols):
        """Build and evaluate multiple ML models"""
        print("\n" + "="*60)
        print("🤖 BUILDING MACHINE LEARNING MODELS")
        print("(Multi-objective: training separate predictive models for cost efficiency and environmental performance)")
        print("="*60)
        
        models = {}
        results = {}
        
        # Define models to test
        model_configs = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(random_state=42)
        }
        
        # Test each target variable
        targets = {
            'operational_efficiency': y_operational,
            'environmental_impact': y_environmental,
            'cost_efficiency': y_cost
        }
        
        for target_name, y in targets.items():
            if y is not None:
                print(f"\n🎯 Building models for {target_name}...")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                target_results = {}
                
                for model_name, model in model_configs.items():
                    try:
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Cross-validation score
                        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                        
                        # Store results
                        target_results[model_name] = {
                            'model': model,
                            'mse': mse,
                            'rmse': rmse,
                            'r2': r2,
                            'cv_mean': cv_mean,
                            'cv_std': cv_std,
                            'y_pred': y_pred,
                            'y_test': y_test
                        }
                        
                        print(f"  ✅ {model_name}: R² = {r2:.3f}, CV R² = {cv_mean:.3f} ± {cv_std:.3f}")
                        
                    except Exception as e:
                        print(f"  ❌ {model_name}: Error - {e}")
                
                # Find best model for this target
                if target_results:
                    best_model = max(target_results.items(), key=lambda x: x[1]['cv_mean'])
                    print(f"  🏆 Best Model: {best_model[0]} (CV R² = {best_model[1]['cv_mean']:.3f})")
                    
                    # Store best model
                    models[f'{target_name}_best'] = best_model[1]['model']
                    results[target_name] = target_results
        
        self.models = models
        self.results = results
        
        return models, results
    
    def feature_importance_analysis(self, X, feature_cols):
        """Analyze feature importance for the best models"""
        print("\n" + "="*60)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        importance_plots = {}
        
        for target_name, target_results in self.results.items():
            if target_results:
                # Get best model
                best_model_name = max(target_results.items(), key=lambda x: x[1]['cv_mean'])[0]
                best_model = target_results[best_model_name]['model']
                
                # Get feature importance
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Create feature importance plot
                    plt.figure(figsize=(12, 8))
                    top_features = feature_importance_df.head(15)
                    plt.barh(range(len(top_features)), top_features['importance'])
                    plt.yticks(range(len(top_features)), top_features['feature'])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top 15 Feature Importance for {target_name.replace("_", " ").title()}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(f'{self.output_dir}/feature_importance_{target_name}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    importance_plots[target_name] = feature_importance_df
                    print(f"✅ Feature importance plot saved for {target_name}")
                    
                    # Print top features
                    print(f"\n📊 Top 5 Features for {target_name}:")
                    for idx, row in feature_importance_df.head().iterrows():
                        print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_plots
    
    def create_optimization_scenarios(self, X, feature_cols):
        """Create optimization scenarios for greener supply chains"""
        print("\n" + "="*60)
        print("🌱 CREATING OPTIMIZATION SCENARIOS")
        print("(Multi-objective lens: quantify improvements in both cost efficiency and sustainability)")
        print("="*60)
        
        scenarios = {}
        
        # Scenario 1: Increase AI & Blockchain Adoption
        if 'has_ai' in feature_cols and 'has_blockchain' in feature_cols:
            print("\n🔄 Scenario 1: Increase AI & Blockchain Adoption")
            
            # Current state
            current_ai = X['has_ai'].mean()
            current_blockchain = X['has_blockchain'].mean()
            
            print(f"  Current AI adoption: {current_ai:.1%}")
            print(f"  Current Blockchain adoption: {current_blockchain:.1%}")
            
            # Optimized state (increase by 20%)
            optimized_ai = min(1.0, current_ai * 1.2)
            optimized_blockchain = min(1.0, current_blockchain * 1.2)
            
            print(f"  Target AI adoption: {optimized_ai:.1%}")
            print(f"  Target Blockchain adoption: {optimized_blockchain:.1%}")
            
            scenarios['ai_blockchain_optimization'] = {
                'current': {'ai': current_ai, 'blockchain': current_blockchain},
                'target': {'ai': optimized_ai, 'blockchain': optimized_blockchain},
                'improvement': {'ai': (optimized_ai - current_ai) / current_ai * 100,
                              'blockchain': (optimized_blockchain - current_blockchain) / current_blockchain * 100}
            }
        
        # Scenario 2: SCM Practice Optimization
        if 'scm_practices_encoded' in feature_cols:
            print("\n🔄 Scenario 2: SCM Practice Optimization")
            
            # Analyze current SCM practices
            scm_practices = self.df['scm_practices'].value_counts()
            sustainable_practices = ['Sustainable SCM', 'Agile SCM', 'Lean Manufacturing']
            
            current_sustainable = sum(scm_practices.get(practice, 0) for practice in sustainable_practices)
            current_sustainable_pct = current_sustainable / len(self.df)
            
            print(f"  Current sustainable practices: {current_sustainable_pct:.1%}")
            
            # Target: Increase sustainable practices by 30%
            target_sustainable_pct = min(1.0, current_sustainable_pct * 1.3)
            
            print(f"  Target sustainable practices: {target_sustainable_pct:.1%}")
            
            scenarios['scm_practice_optimization'] = {
                'current': current_sustainable_pct,
                'target': target_sustainable_pct,
                'improvement': (target_sustainable_pct - current_sustainable_pct) / current_sustainable_pct * 100
            }
        
        # Scenario 3: Supplier Collaboration Enhancement
        if 'supplier_collaboration_level' in self.df.columns:
            print("\n🔄 Scenario 3: Supplier Collaboration Enhancement")
            
            collab_levels = self.df['supplier_collaboration_level'].value_counts()
            high_collab = collab_levels.get('High', 0)
            current_high_pct = high_collab / len(self.df)
            
            print(f"  Current high collaboration: {current_high_pct:.1%}")
            
            # Target: Increase high collaboration by 25%
            target_high_pct = min(1.0, current_high_pct * 1.25)
            
            print(f"  Target high collaboration: {target_high_pct:.1%}")
            
            scenarios['supplier_collaboration_optimization'] = {
                'current': current_high_pct,
                'target': target_high_pct,
                'improvement': (target_high_pct - current_high_pct) / current_high_pct * 100
            }
        
        return scenarios
    
    def generate_optimization_recommendations(self, scenarios, importance_plots):
        """Generate comprehensive optimization recommendations"""
        print("\n" + "="*60)
        print("💡 GENERATING OPTIMIZATION RECOMMENDATIONS")
        print("(Explicit multi-objective framing: recommend actions that improve both cost efficiency and sustainability)")
        print("="*60)
        
        recommendations = []
        recommendations.append("=" * 80)
        recommendations.append("SCM GREEN LOGISTICS - ML OPTIMIZATION RECOMMENDATIONS")
        recommendations.append("=" * 80)
        recommendations.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        recommendations.append("")
        recommendations.append("Objective: Balance cost and carbon outcomes via multi-objective optimization.")
        recommendations.append("")
        
        # Model Performance Summary
        recommendations.append("🤖 ML MODEL PERFORMANCE SUMMARY:")
        for target_name, target_results in self.results.items():
            if target_results:
                best_model_name = max(target_results.items(), key=lambda x: x[1]['cv_mean'])[0]
                best_performance = target_results[best_model_name]['cv_mean']
                recommendations.append(f"  • {target_name.replace('_', ' ').title()}: {best_model_name} (CV R² = {best_performance:.3f})")
        recommendations.append("")
        
        # Feature Importance Insights
        recommendations.append("🔍 KEY FEATURE INSIGHTS:")
        for target_name, importance_df in importance_plots.items():
            top_feature = importance_df.iloc[0]
            recommendations.append(f"  • {target_name.replace('_', ' ').title()}: {top_feature['feature']} is most important")
        recommendations.append("")
        
        # Optimization Scenarios
        recommendations.append("🌱 OPTIMIZATION SCENARIOS (Cost vs Carbon):")
        for scenario_name, scenario_data in scenarios.items():
            if 'improvement' in scenario_data:
                if isinstance(scenario_data['improvement'], dict):
                    for metric, improvement in scenario_data['improvement'].items():
                        recommendations.append(f"  • {scenario_name.replace('_', ' ').title()}: {metric.title()} improvement of {improvement:.1f}%")
                else:
                    recommendations.append(f"  • {scenario_name.replace('_', ' ').title()}: {scenario_data['improvement']:.1f}% improvement")
        recommendations.append("")
        
        # Strategic Recommendations
        recommendations.append("💡 STRATEGIC RECOMMENDATIONS (Multi-objective):")
        recommendations.append("  1. Scale AI + Blockchain to improve efficiency while enabling traceability and waste reduction")
        recommendations.append("  2. Shift to sustainable SCM practices (Lean/Agile/Sustainable) to cut cost and carbon together")
        recommendations.append("  3. Enhance supplier collaboration to stabilize lead times and reduce expediting emissions")
        recommendations.append("  4. Optimize transport mix and load consolidation to lower both OPEX and CO2e per ton-km")
        recommendations.append("  5. Institutionalize data-driven continuous optimization across cost & sustainability KPIs")
        recommendations.append("")
        
        # Implementation Roadmap
        recommendations.append("🚀 IMPLEMENTATION ROADMAP:")
        recommendations.append("  Phase 1 (0-3 months): Multi-objective pilots (cost vs carbon) in 1-2 lanes/products")
        recommendations.append("  Phase 2 (3-6 months): Scale SCM practice optimization and supplier collaboration")
        recommendations.append("  Phase 3 (6-12 months): Expand AI/ERP integration; evaluate blockchain for compliance")
        recommendations.append("  Phase 4 (12+ months): Continuous optimization & carbon accounting integration")
        
        # Save recommendations
        with open(f'{self.output_dir}/ml_optimization_recommendations.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(recommendations))
        
        print("✅ Optimization recommendations saved")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("ML OPTIMIZATION RECOMMENDATIONS SUMMARY")
        print("=" * 80)
        for line in recommendations:
            print(line)
    
    def create_ml_visualizations(self):
        """Create ML model performance visualizations"""
        print("\n" + "="*60)
        print("🎨 CREATING ML VISUALIZATIONS")
        print("="*60)
        
        # Create model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SCM Green Logistics - ML Model Performance Overview', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Comparison
        if self.results:
            target_names = list(self.results.keys())
            if target_names:
                target_name = target_names[0]
                target_results = self.results[target_name]
                
                model_names = list(target_results.keys())
                cv_scores = [target_results[model]['cv_mean'] for model in model_names]
                cv_stds = [target_results[model]['cv_std'] for model in model_names]
                
                axes[0, 0].bar(model_names, cv_scores, yerr=cv_stds, capsize=5, color='skyblue', edgecolor='black')
                axes[0, 0].set_title(f'Model Performance for {target_name.replace("_", " ").title()}', fontweight='bold')
                axes[0, 0].set_ylabel('Cross-Validation R² Score')
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Feature Importance (if available)
        if hasattr(self, 'importance_plots') and self.importance_plots:
            target_name = list(self.importance_plots.keys())[0]
            importance_df = self.importance_plots[target_name].head(10)
            
            axes[0, 1].barh(range(len(importance_df)), importance_df['importance'])
            axes[0, 1].set_yticks(range(len(importance_df)))
            axes[0, 1].set_yticklabels(importance_df['feature'])
            axes[0, 1].set_title(f'Top 10 Feature Importance for {target_name.replace("_", " ").title()}', fontweight='bold')
            axes[0, 1].set_xlabel('Feature Importance')
            axes[0, 1].invert_yaxis()
        
        # 3. Prediction vs Actual (if available)
        if self.results:
            target_name = list(self.results.keys())[0]
            target_results = self.results[target_name]
            best_model_name = max(target_results.items(), key=lambda x: x[1]['cv_mean'])[0]
            best_result = target_results[best_model_name]
            
            axes[1, 0].scatter(best_result['y_test'], best_result['y_pred'], alpha=0.6, s=30)
            axes[1, 0].plot([best_result['y_test'].min(), best_result['y_test'].max()], 
                           [best_result['y_test'].min(), best_result['y_test'].max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual Values')
            axes[1, 0].set_ylabel('Predicted Values')
            axes[1, 0].set_title(f'Prediction vs Actual for {target_name.replace("_", " ").title()}', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Optimization Impact
        if hasattr(self, 'scenarios') and self.scenarios:
            scenario_names = list(self.scenarios.keys())
            improvements = []
            
            for scenario_name in scenario_names:
                scenario_data = self.scenarios[scenario_name]
                if 'improvement' in scenario_data:
                    if isinstance(scenario_data['improvement'], dict):
                        avg_improvement = np.mean(list(scenario_data['improvement'].values()))
                    else:
                        avg_improvement = scenario_data['improvement']
                    improvements.append(avg_improvement)
            
            if improvements:
                axes[1, 1].bar(range(len(scenario_names)), improvements, color='lightgreen', edgecolor='black')
                axes[1, 1].set_title('Optimization Scenario Impact', fontweight='bold')
                axes[1, 1].set_ylabel('Expected Improvement (%)')
                axes[1, 1].set_xticks(range(len(scenario_names)))
                axes[1, 1].set_xticklabels([name.replace('_', ' ').title() for name in scenario_names], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ml_model_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ ML model overview visualization saved")
    
    def run_full_ml_optimization(self):
        """Run the complete ML optimization pipeline"""
        print("🚀 Starting SCM Green Logistics ML Optimization")
        print("(Goal: Multi-objective optimization balancing cost efficiency and carbon performance)")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Prepare features
        X, y_operational, y_environmental, y_cost, feature_cols = self.prepare_features()
        
        # Build ML models
        models, results = self.build_ml_models(X, y_operational, y_environmental, y_cost, feature_cols)
        
        # Analyze feature importance
        importance_plots = self.feature_importance_analysis(X, feature_cols)
        
        # Create optimization scenarios
        scenarios = self.create_optimization_scenarios(X, feature_cols)
        
        # Generate recommendations
        self.generate_optimization_recommendations(scenarios, importance_plots)
        
        # Create visualizations
        self.create_ml_visualizations()
        
        print("\n" + "=" * 70)
        print("✅ ML OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 All outputs saved in: {self.output_dir}/")
        print("🤖 ML models built and evaluated")
        print("🌱 Optimization scenarios created")
        print("💡 Recommendations generated (multi-objective)")
        print("🎯 Ready for Phase 5: LLM Integration for Insights")
        
        return True

def main():
    """Main execution function"""
    ml_optimization = SCMMLOptimization()
    success = ml_optimization.run_full_ml_optimization()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Review ML outputs in ml_outputs/ folder")
        print("2. Examine optimization recommendations (cost vs carbon trade-off)")
        print("3. Proceed to Phase 5: LLM Integration for Insights")
    else:
        print("\n❌ ML optimization failed. Please check the errors above.")

if __name__ == "__main__":
    main()
