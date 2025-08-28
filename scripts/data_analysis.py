#!/usr/bin/env python3
"""
Cucumber Trait Data Analysis
Comprehensive analysis of extracted scientific traits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CucumberTraitAnalyzer:
    def __init__(self, csv_path):
        """Initialize the analyzer with the trait data."""
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path(csv_path).parent / 'analysis_plots'
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"📊 Loaded {len(self.df)} cucumber samples")
        print(f"📁 Output directory: {self.output_dir}")
    
    def basic_statistics(self):
        """Generate basic descriptive statistics."""
        print("\n📈 BASIC STATISTICS")
        print("=" * 50)
        
        # Overall statistics
        print(f"Total cucumbers analyzed: {len(self.df)}")
        print(f"Images processed: {self.df['image_path'].nunique()}")
        print(f"Classes found: {self.df['class'].value_counts().to_dict()}")
        
        # Mask quality analysis
        sam2_count = (self.df['mask_source'] == 'SAM2').sum()
        generated_count = (self.df['mask_source'] == 'Generated').sum()
        print(f"\nMask Sources:")
        print(f"  SAM2 (High Quality): {sam2_count} ({sam2_count/len(self.df)*100:.1f}%)")
        print(f"  Generated: {generated_count} ({generated_count/len(self.df)*100:.1f}%)")
        
        # Confidence analysis
        print(f"\nDetection Confidence:")
        print(f"  Average: {self.df['confidence'].mean():.3f}")
        print(f"  Range: {self.df['confidence'].min():.3f} - {self.df['confidence'].max():.3f}")
        
        return self.df.describe()
    
    def trait_distributions(self):
        """Plot distributions of key traits."""
        print("\n📊 PLOTTING TRAIT DISTRIBUTIONS")
        print("=" * 50)
        
        # Key traits to analyze
        key_traits = [
            'curved_length_cm', 'diameter_cm', 'fruit_shape_index',
            'hollowness_percentage', 'curvature_ratio', 'tapering_ratio',
            'netting_score', 'spine_density_per_cm2'
        ]
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Distribution of Cucumber Traits', fontsize=16, fontweight='bold')
        
        for i, trait in enumerate(key_traits):
            if trait in self.df.columns:
                row = i // 4
                col = i % 4
                
                # Plot histogram
                axes[row, col].hist(self.df[trait].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'{trait.replace("_", " ").title()}')
                axes[row, col].set_xlabel(trait.replace('_', ' ').title())
                axes[row, col].set_ylabel('Frequency')
                
                # Add statistics
                mean_val = self.df[trait].mean()
                std_val = self.df[trait].std()
                axes[row, col].axvline(mean_val, color='red', linestyle='--', 
                                     label=f'Mean: {mean_val:.2f}')
                axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trait_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved trait distributions to: {self.output_dir / 'trait_distributions.png'}")
    
    def trait_correlations(self):
        """Analyze correlations between traits."""
        print("\n🔗 ANALYZING TRAIT CORRELATIONS")
        print("=" * 50)
        
        # Select numeric traits
        numeric_traits = self.df.select_dtypes(include=[np.number]).columns
        numeric_traits = [col for col in numeric_traits if col not in ['confidence', 'mask_quality_score']]
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_traits].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Cucumber Traits', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trait_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find strongest correlations
        print("\n🔍 Strongest Correlations (|r| > 0.5):")
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        # Sort by absolute correlation value
        strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for trait1, trait2, corr_val in strong_corrs[:10]:
            print(f"  {trait1} ↔ {trait2}: r = {corr_val:.3f}")
        
        print(f"\n✅ Saved correlation matrix to: {self.output_dir / 'trait_correlations.png'}")
        return corr_matrix
    
    def class_comparison(self):
        """Compare traits between different cucumber classes."""
        print("\n🥒 COMPARING CUCUMBER CLASSES")
        print("=" * 50)
        
        if 'class' not in self.df.columns:
            print("No class information available")
            return
        
        # Get unique classes
        classes = self.df['class'].unique()
        print(f"Classes found: {classes}")
        
        # Key traits to compare
        comparison_traits = [
            'curved_length_cm', 'diameter_cm', 'fruit_shape_index',
            'hollowness_percentage', 'curvature_ratio', 'tapering_ratio'
        ]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trait Comparison by Cucumber Class', fontsize=16, fontweight='bold')
        
        for i, trait in enumerate(comparison_traits):
            if trait in self.df.columns:
                row = i // 3
                col = i % 3
                
                # Create box plot
                sns.boxplot(data=self.df, x='class', y=trait, ax=axes[row, col])
                axes[row, col].set_title(f'{trait.replace("_", " ").title()}')
                axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical comparison
        print("\n📊 Statistical Comparison (ANOVA):")
        for trait in comparison_traits:
            if trait in self.df.columns:
                # Perform one-way ANOVA
                groups = [group[trait].dropna().values for name, group in self.df.groupby('class')]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    print(f"  {trait}: F = {f_stat:.3f}, p = {p_value:.4f}")
        
        print(f"\n✅ Saved class comparison to: {self.output_dir / 'class_comparison.png'}")
    
    def netting_analysis(self):
        """Analyze netting patterns and relationships."""
        print("\n🌐 NETTING ANALYSIS")
        print("=" * 50)
        
        if 'netting_score' not in self.df.columns:
            print("No netting data available")
            return
        
        # Netting distribution
        netting_counts = self.df['netting_score'].value_counts().sort_index()
        print("Netting Score Distribution:")
        for score, count in netting_counts.items():
            percentage = count / len(self.df) * 100
            description = self.df[self.df['netting_score'] == score]['netting_description'].iloc[0]
            print(f"  Score {score} ({description}): {count} cucumbers ({percentage:.1f}%)")
        
        # Plot netting distribution
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Netting score distribution
        plt.subplot(2, 2, 1)
        netting_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Distribution of Netting Scores')
        plt.xlabel('Netting Score')
        plt.ylabel('Number of Cucumbers')
        plt.xticks(rotation=0)
        
        # Subplot 2: Netting vs Length
        plt.subplot(2, 2, 2)
        sns.boxplot(data=self.df, x='netting_score', y='curved_length_cm')
        plt.title('Netting Score vs Fruit Length')
        plt.xlabel('Netting Score')
        plt.ylabel('Length (cm)')
        
        # Subplot 3: Netting vs Diameter
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.df, x='netting_score', y='diameter_cm')
        plt.title('Netting Score vs Fruit Diameter')
        plt.xlabel('Netting Score')
        plt.ylabel('Diameter (cm)')
        
        # Subplot 4: Netting vs Shape Index
        plt.subplot(2, 2, 4)
        sns.boxplot(data=self.df, x='netting_score', y='fruit_shape_index')
        plt.title('Netting Score vs Fruit Shape Index')
        plt.xlabel('Netting Score')
        plt.ylabel('Shape Index (L/D)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'netting_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Saved netting analysis to: {self.output_dir / 'netting_analysis.png'}")
    
    def color_analysis(self):
        """Analyze color patterns and relationships."""
        print("\n🎨 COLOR ANALYSIS")
        print("=" * 50)
        
        color_cols = ['avg_red', 'avg_green', 'avg_blue']
        if not all(col in self.df.columns for col in color_cols):
            print("No color data available")
            return
        
        # Basic color statistics
        print("Color Statistics (RGB values):")
        for col in color_cols:
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            print(f"  {col}: Mean = {mean_val:.1f}, Std = {std_val:.1f}")
        
        # Create color analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cucumber Color Analysis', fontsize=16, fontweight='bold')
        
        # RGB distributions
        for i, col in enumerate(color_cols):
            axes[0, i].hist(self.df[col].dropna(), bins=30, alpha=0.7, 
                           color=col.replace('avg_', ''), edgecolor='black')
            axes[0, i].set_title(f'{col.replace("_", " ").title()} Distribution')
            axes[0, i].set_xlabel('RGB Value')
            axes[0, i].set_ylabel('Frequency')
        
        # Color relationships
        # Red vs Green
        axes[1, 0].scatter(self.df['avg_red'], self.df['avg_green'], alpha=0.6)
        axes[1, 0].set_xlabel('Red')
        axes[1, 0].set_ylabel('Green')
        axes[1, 0].set_title('Red vs Green')
        
        # Red vs Blue
        axes[1, 1].scatter(self.df['avg_red'], self.df['avg_blue'], alpha=0.6)
        axes[1, 1].set_xlabel('Red')
        axes[1, 1].set_ylabel('Blue')
        axes[1, 1].set_title('Red vs Blue')
        
        # Green vs Blue
        axes[1, 2].scatter(self.df['avg_green'], self.df['avg_blue'], alpha=0.6)
        axes[1, 2].set_xlabel('Green')
        axes[1, 2].set_ylabel('Blue')
        axes[1, 2].set_title('Green vs Blue')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'color_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Saved color analysis to: {self.output_dir / 'color_analysis.png'}")
    
    def quality_analysis(self):
        """Analyze data quality and mask performance."""
        print("\n🔍 DATA QUALITY ANALYSIS")
        print("=" * 50)
        
        # Mask source comparison
        if 'mask_source' in self.df.columns:
            print("Mask Source Performance:")
            for source in self.df['mask_source'].unique():
                source_data = self.df[self.df['mask_source'] == source]
                print(f"\n  {source} Masks ({len(source_data)} cucumbers):")
                
                # Compare key traits
                for trait in ['curved_length_cm', 'diameter_cm', 'fruit_shape_index']:
                    if trait in source_data.columns:
                        mean_val = source_data[trait].mean()
                        std_val = source_data[trait].std()
                        print(f"    {trait}: {mean_val:.2f} ± {std_val:.2f}")
        
        # Confidence analysis
        if 'confidence' in self.df.columns:
            print(f"\nDetection Confidence Analysis:")
            print(f"  High confidence (>0.9): {(self.df['confidence'] > 0.9).sum()} cucumbers")
            print(f"  Medium confidence (0.7-0.9): {((self.df['confidence'] >= 0.7) & (self.df['confidence'] <= 0.9)).sum()} cucumbers")
            print(f"  Low confidence (<0.7): {(self.df['confidence'] < 0.7).sum()} cucumbers")
        
        # Missing data analysis
        print(f"\nMissing Data Analysis:")
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                percentage = missing_count / len(self.df) * 100
                print(f"  {col}: {missing_count} missing values ({percentage:.1f}%)")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n📋 GENERATING SUMMARY REPORT")
        print("=" * 50)
        
        report_path = self.output_dir / 'cucumber_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("CUCUMBER TRAIT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write(f"Total cucumbers analyzed: {len(self.df)}\n")
            f.write(f"Images processed: {self.df['image_path'].nunique()}\n")
            f.write(f"Classes found: {self.df['class'].value_counts().to_dict()}\n\n")
            
            # Trait summaries
            f.write("TRAIT SUMMARIES:\n")
            f.write("-" * 20 + "\n")
            
            numeric_traits = self.df.select_dtypes(include=[np.number]).columns
            numeric_traits = [col for col in numeric_traits if col not in ['confidence', 'mask_quality_score']]
            
            for trait in numeric_traits:
                if trait in self.df.columns:
                    data = self.df[trait].dropna()
                    if len(data) > 0:
                        f.write(f"{trait}:\n")
                        f.write(f"  Count: {len(data)}\n")
                        f.write(f"  Mean: {data.mean():.3f}\n")
                        f.write(f"  Std: {data.std():.3f}\n")
                        f.write(f"  Min: {data.min():.3f}\n")
                        f.write(f"  Max: {data.max():.3f}\n")
                        f.write(f"  Median: {data.median():.3f}\n\n")
            
            # Quality metrics
            f.write("QUALITY METRICS:\n")
            f.write("-" * 20 + "\n")
            
            if 'mask_source' in self.df.columns:
                sam2_count = (self.df['mask_source'] == 'SAM2').sum()
                f.write(f"SAM2 masks: {sam2_count} ({sam2_count/len(self.df)*100:.1f}%)\n")
            
            if 'confidence' in self.df.columns:
                f.write(f"Average confidence: {self.df['confidence'].mean():.3f}\n")
                f.write(f"High confidence (>0.9): {(self.df['confidence'] > 0.9).sum()}\n")
        
        print(f"✅ Summary report saved to: {report_path}")
    
    def run_complete_analysis(self):
        """Run all analysis components."""
        print("🚀 STARTING COMPLETE CUCUMBER TRAIT ANALYSIS")
        print("=" * 60)
        
        # Run all analyses
        self.basic_statistics()
        self.trait_distributions()
        self.trait_correlations()
        self.class_comparison()
        self.netting_analysis()
        self.color_analysis()
        self.quality_analysis()
        self.generate_summary_report()
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 All results saved to: {self.output_dir}")
        print("📊 Generated visualizations and statistical summaries")

def main():
    parser = argparse.ArgumentParser(description='Cucumber Trait Data Analyzer')
    parser.add_argument('--csv-path', required=True, help='Path to the comprehensive traits CSV file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CucumberTraitAnalyzer(args.csv_path)
    
    # Run complete analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
