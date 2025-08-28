#!/usr/bin/env python3
"""
Create HTML Viewer for Cucumber Trait Visualizations
"""

import pandas as pd
from pathlib import Path
import argparse

def create_html_viewer(csv_path, visualization_dir, output_html):
    """Create an HTML viewer showing images with trait data."""
    
    # Load the trait data
    df = pd.read_csv(csv_path)
    
    # Get visualization images
    vis_dir = Path(visualization_dir)
    image_files = list(vis_dir.glob('*_traits.jpg'))
    
    # Group data by image
    image_groups = df.groupby('image_path')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cucumber Trait Visualizations</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .image-section {{ 
                background: white; 
                margin: 20px 0; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .image-container {{ text-align: center; margin: 20px 0; }}
            .image-container img {{ max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 5px; }}
            .trait-table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0;
                background: white;
            }}
            .trait-table th, .trait-table td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }}
            .trait-table th {{ 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold;
            }}
            .trait-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .header {{ 
                text-align: center; 
                color: #2E7D32; 
                margin: 30px 0;
                font-size: 2.5em;
                font-weight: bold;
            }}
            .subheader {{ 
                text-align: center; 
                color: #388E3C; 
                margin: 20px 0;
                font-size: 1.5em;
            }}
            .stats {{ 
                background: #E8F5E8; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 20px 0;
                border-left: 5px solid #4CAF50;
            }}
        </style>
    </head>
    <body>
        <div class="header">🥒 Cucumber Trait Analysis Results</div>
        <div class="subheader">Scientific Trait Extraction from Images</div>
        
        <div class="stats">
            <h3>📊 Overall Statistics</h3>
            <p><strong>Total cucumbers analyzed:</strong> {len(df)}</p>
            <p><strong>Images processed:</strong> {len(image_groups)}</p>
            <p><strong>Classes found:</strong> {', '.join(df['class'].unique())}</p>
            <p><strong>High-quality SAM2 masks:</strong> {(df['mask_source'] == 'SAM2').mean() * 100:.1f}%</p>
        </div>
    """
    
    # Add each image section
    for i, (image_path, group_data) in enumerate(image_groups):
        # Find the corresponding visualization image
        image_name = Path(image_path).name
        vis_image = None
        
        for img_file in image_files:
            if image_name.replace('.rf.', '_rf_') in img_file.name or image_name.split('.rf.')[0] in img_file.name:
                vis_image = img_file
                break
        
        if vis_image:
            html_content += f"""
            <div class="image-section">
                <h2>📸 Image {i+1}: {image_name}</h2>
                <p><strong>Cucumbers detected:</strong> {len(group_data)}</p>
                
                <div class="image-container">
                    <img src="{vis_image}" alt="Cucumber traits visualization">
                </div>
                
                <h3>🔍 Extracted Traits for Each Cucumber:</h3>
                <table class="trait-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Class</th>
                            <th>Confidence</th>
                            <th>Length (cm)</th>
                            <th>Diameter (cm)</th>
                            <th>Shape Index</th>
                            <th>Hollowness (%)</th>
                            <th>Netting</th>
                            <th>Spine Density (/cm²)</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for idx, cucumber in group_data.iterrows():
                html_content += f"""
                        <tr>
                            <td>{idx}</td>
                            <td>{cucumber['class']}</td>
                            <td>{cucumber['confidence']:.3f}</td>
                            <td>{cucumber.get('curved_length_cm', 'N/A'):.1f}</td>
                            <td>{cucumber.get('diameter_cm', 'N/A'):.1f}</td>
                            <td>{cucumber.get('fruit_shape_index', 'N/A'):.2f}</td>
                            <td>{cucumber.get('hollowness_percentage', 'N/A'):.1f}</td>
                            <td>{cucumber.get('netting_description', 'N/A')}</td>
                            <td>{cucumber.get('spine_density_per_cm2', 'N/A'):.1f}</td>
                        </tr>
                """
            
            html_content += """
                    </tbody>
                </table>
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML viewer created: {output_html}")
    print(f"🌐 Open this file in your web browser to view the results")

def main():
    parser = argparse.ArgumentParser(description='Create HTML Viewer for Cucumber Traits')
    parser.add_argument('--csv-path', default='results/scientific_traits_test/csv_reports/comprehensive_cucumber_traits.csv',
                       help='Path to the comprehensive traits CSV file')
    parser.add_argument('--visualization-dir', default='results/trait_visualizations',
                       help='Directory containing visualization images')
    parser.add_argument('--output-html', default='results/cucumber_traits_viewer.html',
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    create_html_viewer(args.csv_path, args.visualization_dir, args.output_html)

if __name__ == "__main__":
    main()
