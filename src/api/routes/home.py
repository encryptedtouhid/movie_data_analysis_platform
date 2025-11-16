"""
Home Page Route
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from src.core.config import settings

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home():
    """
    Home page with API interface
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Data Analysis Platform</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: #ffffff;
            color: #24292f;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1280px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        header {{
            margin-bottom: 48px;
        }}
        header h1 {{
            font-size: 40px;
            font-weight: 600;
            color: #24292f;
            margin-bottom: 12px;
        }}
        header p {{
            font-size: 18px;
            color: #57606a;
        }}
        .section {{
            margin-bottom: 48px;
        }}
        .section h2 {{
            font-size: 24px;
            font-weight: 600;
            color: #24292f;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid #d0d7de;
        }}
        .endpoint-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .endpoint-card {{
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 16px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .endpoint-card:hover {{
            border-color: #0969da;
            box-shadow: 0 3px 8px rgba(9, 105, 218, 0.12);
        }}
        .endpoint-card h3 {{
            font-size: 16px;
            font-weight: 600;
            color: #24292f;
            margin-bottom: 8px;
        }}
        .endpoint-card p {{
            font-size: 14px;
            color: #57606a;
            margin-bottom: 12px;
        }}
        .endpoint-card .method {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }}
        .method.get {{
            background: #dafbe1;
            color: #1a7f37;
        }}
        .method.post {{
            background: #ddf4ff;
            color: #0969da;
        }}
        .endpoint-path {{
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: #57606a;
        }}
        button {{
            background: #0969da;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #0860ca;
        }}
        button:disabled {{
            background: #94a3b8;
            cursor: not-allowed;
        }}
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        .modal-overlay.active {{
            display: flex;
        }}
        .modal-container {{
            background: white;
            border-radius: 8px;
            width: 90%;
            max-width: 1200px;
            max-height: 90vh;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
        }}
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #d0d7de;
        }}
        .modal-header h3 {{
            font-size: 20px;
            font-weight: 600;
            margin: 0;
        }}
        .close-btn {{
            background: #d0d7de;
            color: #24292f;
            padding: 6px 16px;
            border-radius: 6px;
            cursor: pointer;
            border: none;
            font-size: 14px;
            font-weight: 500;
        }}
        .close-btn:hover {{
            background: #afb8c1;
        }}
        .modal-content {{
            padding: 20px;
            overflow: auto;
            flex: 1;
        }}
        .result-content {{
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
        }}
        .result-content img {{
            max-width: 100%;
            border-radius: 6px;
        }}
        .result-content iframe {{
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 6px;
        }}
        .result-content pre {{
            background: #24292f;
            color: #c9d1d9;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
        }}
        .loading {{
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #d0d7de;
            border-top-color: #0969da;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
            margin-left: 8px;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        .footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 32px 0;
            border-top: 1px solid #d0d7de;
            margin-top: 48px;
            color: #57606a;
            font-size: 14px;
        }}
        .footer a {{
            color: #0969da;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Movie Data Analysis Platform</h1>
            <p>Interactive API Dashboard</p>
        </header>

        <div class="section">
            <h2>Data Processing Endpoints</h2>
            <div class="endpoint-grid">
                <div class="endpoint-card" onclick="callEndpoint('/api/v1/dataprocess/process_all_data', 'POST')">
                    <h3>Process All Data</h3>
                    <p>Convert, load, clean, and analyze all datasets</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/dataprocess/process_all_data</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/dataprocess/load_data', 'POST', {{dataset: 'ratings'}})">
                    <h3>Load Data</h3>
                    <p>Load specific dataset (movies, ratings, tags, users)</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/dataprocess/load_data</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/dataprocess/clean_data', 'POST', {{dataset: 'ratings'}})">
                    <h3>Clean Data</h3>
                    <p>Remove duplicates and handle missing values</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/dataprocess/clean_data</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/dataprocess/aggregate_statistics', 'POST', {{dataset: 'all'}})">
                    <h3>Aggregate Statistics</h3>
                    <p>Calculate comprehensive statistics for datasets</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/dataprocess/aggregate_statistics</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/dataprocess/filter_data', 'POST', {{dataset: 'ratings', min_rating: 4.0, limit: 100}})">
                    <h3>Filter Data</h3>
                    <p>Apply filters to datasets (rating, genre, year, etc.)</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/dataprocess/filter_data</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/dataprocess/export_data', 'POST', {{dataset: 'ratings', format: 'json', limit: 50}})">
                    <h3>Export Data</h3>
                    <p>Export datasets to CSV or JSON format</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/dataprocess/export_data</span>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Analysis Endpoints</h2>
            <div class="endpoint-grid">
                <div class="endpoint-card" onclick="callEndpoint('/api/v1/analysis/top_movies', 'POST', {{limit: 20, min_ratings: 50}})">
                    <h3>Top Movies</h3>
                    <p>Get highest-rated movies with statistical significance</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/analysis/top_movies</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/analysis/genre_trends', 'GET')">
                    <h3>Genre Trends</h3>
                    <p>Analyze popularity and rating trends across genres</p>
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/api/v1/analysis/genre_trends</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/analysis/time_series', 'GET')">
                    <h3>Time Series</h3>
                    <p>Analyze rating patterns over time</p>
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/api/v1/analysis/time_series</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/analysis/correlation_analysis', 'GET')">
                    <h3>Correlation Analysis</h3>
                    <p>Analyze correlations between movie metrics</p>
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/api/v1/analysis/correlation_analysis</span>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Visualizations</h2>
            <div class="endpoint-grid">
                <div class="endpoint-card" onclick="callVisualization('rating_distribution')">
                    <h3>Rating Distribution</h3>
                    <p>Histogram and count plot of ratings</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/analysis/visualize</span>
                </div>

                <div class="endpoint-card" onclick="callVisualization('genre_popularity')">
                    <h3>Genre Popularity</h3>
                    <p>Genre popularity and rating charts</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/analysis/visualize</span>
                </div>

                <div class="endpoint-card" onclick="callVisualization('time_series')">
                    <h3>Time Series Plot</h3>
                    <p>Temporal trends visualization</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/analysis/visualize</span>
                </div>

                <div class="endpoint-card" onclick="callVisualization('dashboard')">
                    <h3>Dashboard Report</h3>
                    <p>Comprehensive HTML dashboard</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/analysis/visualize</span>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🤖 ML Recommendations</h2>
            <div class="endpoint-grid">
                <div class="endpoint-card" onclick="callEndpoint('/api/v1/recommendations/similar_movies', 'POST', {{movie_id: 1, limit: 10, min_common_ratings: 50}})">
                    <h3>Similar Movies</h3>
                    <p>Content-based filtering: Find movies similar to a given movie</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/recommendations/similar_movies</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/recommendations/user_recommendations', 'POST', {{user_id: 1, limit: 10, min_user_overlap: 50}})">
                    <h3>User Recommendations</h3>
                    <p>Collaborative filtering: Personalized movie recommendations</p>
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/v1/recommendations/user_recommendations</span>
                </div>

                <div class="endpoint-card" onclick="callEndpoint('/api/v1/recommendations/status', 'GET')">
                    <h3>Recommender Status</h3>
                    <p>Check ML recommendation engine initialization status</p>
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/api/v1/recommendations/status</span>
                </div>
            </div>
        </div>

        <div class="footer">
            <div>Movie Data Analysis Platform</div>
            <div>Developed by: <a href="https://github.com/encryptedtouhid" target="_blank">Khaled Md Tuhidul Hossain</a></div>
        </div>
    </div>

    <!-- Modal Popup -->
    <div class="modal-overlay" id="modalOverlay" onclick="closeModal(event)">
        <div class="modal-container" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3 id="modalTitle">Results</h3>
                <button class="close-btn" onclick="closeModal()">Close</button>
            </div>
            <div class="modal-content">
                <div class="result-content" id="resultContent"></div>
            </div>
        </div>
    </div>

    <script>
        function openModal(title = 'Results') {{
            document.getElementById('modalTitle').textContent = title;
            document.getElementById('modalOverlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}

        function closeModal(event) {{
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('modalOverlay').classList.remove('active');
            document.body.style.overflow = 'auto';
        }}

        async function callEndpoint(endpoint, method, body = null) {{
            const resultContent = document.getElementById('resultContent');

            openModal('Loading Results');
            resultContent.innerHTML = '<div>Loading<span class="loading"></span></div>';

            try {{
                const options = {{
                    method: method,
                    headers: {{
                        'Content-Type': 'application/json',
                    }}
                }};

                if (body) {{
                    options.body = JSON.stringify(body);
                }}

                const response = await fetch(endpoint, options);
                const data = await response.json();

                document.getElementById('modalTitle').textContent = 'Results';
                resultContent.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            }} catch (error) {{
                document.getElementById('modalTitle').textContent = 'Error';
                resultContent.innerHTML = '<div style="color: #cf222e;">Error: ' + error.message + '</div>';
            }}
        }}

        async function callVisualization(type) {{
            const resultContent = document.getElementById('resultContent');

            openModal('Generating Visualization');
            resultContent.innerHTML = '<div>Generating visualization<span class="loading"></span></div>';

            try {{
                const response = await fetch('/api/v1/analysis/visualize', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ visualization_type: type }})
                }});

                const data = await response.json();

                if (data.url) {{
                    const titles = {{
                        'dashboard': 'Dashboard Report',
                        'rating_distribution': 'Rating Distribution',
                        'genre_popularity': 'Genre Popularity',
                        'time_series': 'Time Series Analysis'
                    }};
                    document.getElementById('modalTitle').textContent = titles[type] || 'Visualization';

                    if (type === 'dashboard') {{
                        // For HTML dashboard, open in iframe
                        resultContent.innerHTML = '<iframe src="' + data.url + '"></iframe>';
                    }} else {{
                        // For images
                        resultContent.innerHTML = '<img src="' + data.url + '" alt="' + type + '">';
                    }}
                }} else {{
                    document.getElementById('modalTitle').textContent = 'Results';
                    resultContent.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                }}
            }} catch (error) {{
                document.getElementById('modalTitle').textContent = 'Error';
                resultContent.innerHTML = '<div style="color: #cf222e;">Error: ' + error.message + '</div>';
            }}
        }}
    </script>
</body>
</html>
"""
    return html_content
