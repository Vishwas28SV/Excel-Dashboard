Interactive Excel Dashboard Generator
A production-ready Python application that automatically converts Excel files into beautiful, interactive dashboards with zero configuration required.

✨ Features
🎯 Automatic Intelligence
Smart Column Detection: Automatically identifies date, numeric, and categorical columns
Dynamic KPIs: Intelligently selects relevant metrics (totals, averages, unique counts)
Adaptive Visualizations: Generates appropriate charts based on data types

📊 Visualizations
KPI Cards: Key metrics at a glance
Time Series Charts: Trends over time with interactive zoom and pan
Category Distributions: Top 10 bar charts for categorical data
Data Preview: Paginated table view with all data

🔍 Interactive Filtering
Date Range Filters: Select custom date ranges
Categorical Filters: Multi-select dropdowns for categories
Numeric Range Filters: Slider controls for numeric columns
Real-time Updates: All charts update instantly when filters change

💪 Production Quality
Error Handling: Graceful handling of malformed Excel files
Responsive Design: Works on desktop, tablet, and mobile
Performance Optimized: Handles datasets with thousands of rows
Export Capability: Download filtered data as CSV

🚀 Quick Start
Installation
pip install -r requirements.txt

Requirements:
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
openpyxl>=3.1.0
numpy>=1.24.0

Running the Application
streamlit run excel_dashboard.py

📋 Usage
Upload Excel File: Click "Choose an Excel file" and select your .xlsx file
Auto-Analysis: The app automatically detects column types and generates visualizations
Apply Filters: Use the sidebar to filter data by date, category, or numeric ranges
Explore Data: Interact with charts (zoom, pan, hover for details)
Export Results: Download filtered data as CSV
🎯 Supported Data Types
The dashboard automatically handles:

Type	Examples	Auto-Detection
Dates	Order Date, Created At, Timestamp	Parses common date formats
Numeric	Sales Amount, Quantity, Rating	Identifies numbers for metrics
Categorical	Category, Region, Status	Text/mixed values with reasonable cardinality

📊 Example Use Cases
Sales Analytics: Track revenue, orders, and customer metrics over time
Customer Feedback: Analyze ratings, sentiment, and response patterns
Inventory Management: Monitor stock levels, product categories, and locations
Survey Results: Visualize responses across demographics and questions
Transaction Analysis: Examine payment patterns, amounts, and trends

🛠️ Technical Stack
Streamlit: Interactive web UI framework
Pandas: Data manipulation and analysis
Plotly: Interactive, publication-quality charts
OpenPyXL: Excel file reading

