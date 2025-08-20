import os
import re

def add_creator_credit_to_dashboards():
    """Add creator credit to all dashboard HTML files"""
    dashboard_dir = 'dashboard_outputs'
    
    # Creator credit HTML to add before </body>
    creator_credit = '''
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);">
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; font-weight: 500; margin: 0;">
                Created by <span style="color: #ffd700; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">Jay S Kaphale</span> | Advanced Analytics & ML Solutions
            </p>
        </div>
    </body>
</html>'''
    
    # Process each HTML file in dashboard_outputs
    for filename in os.listdir(dashboard_dir):
        if filename.endswith('.html') and filename != 'index.html':  # Skip index.html as it's already updated
            filepath = os.path.join(dashboard_dir, filename)
            print(f"🔄 Processing: {filename}")
            
            try:
                # Read the file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if creator credit already exists
                if 'Created by Jay S Kaphale' in content:
                    print(f"  ✅ Creator credit already exists in {filename}")
                    continue
                
                # Replace the closing body tag with creator credit + body tag
                if '</body>' in content:
                    # Find the last occurrence of </body> and replace it
                    content = re.sub(r'</body>\s*</html>', creator_credit, content)
                    
                    # Write back to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"  ✅ Added creator credit to {filename}")
                else:
                    print(f"  ⚠️ Could not find </body> tag in {filename}")
                    
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
    
    print("\n🎯 Creator credit addition completed!")
    print("📁 Check dashboard_outputs/ folder for updated files")

if __name__ == "__main__":
    add_creator_credit_to_dashboards()
