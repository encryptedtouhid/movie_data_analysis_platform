import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from src.core.config import settings
from src.exceptions import DataAnalysisError
from src.utils.logger import get_logger

logger = get_logger("data_visualizer", "visualization")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DataVisualizer:
    """
    Create visualizations and reports from movie data analysis
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize DataVisualizer

        Args:
            output_dir: Directory to save visualization outputs (default: data/visualizations)
        """
        logger.info("Initializing DataVisualizer")
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(settings.data_processed_path).parent / "visualizations"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizations will be saved to: {self.output_dir}")

    def create_rating_distribution(self, df: pd.DataFrame) -> str:
        """
        Generate rating distribution plot

        Args:
            df: DataFrame with rating data

        Returns:
            Path to saved plot
        """
        try:
            logger.info("Creating rating distribution plot")

            if df.empty or 'rating' not in df.columns:
                raise DataAnalysisError("DataFrame must contain 'rating' column")

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram
            ax1.hist(df['rating'], bins=10, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.set_xlabel('Rating', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Rating Distribution (Histogram)', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Count plot
            rating_counts = df['rating'].value_counts().sort_index()
            ax2.bar(rating_counts.index, rating_counts.values, color='coral', edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Rating', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title('Rating Count by Value', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Generate unique filename with UUID
            unique_id = str(uuid.uuid4())[:8]
            filename = f"rating_distribution_{unique_id}.png"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Rating distribution plot saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create rating distribution plot: {str(e)}")
            raise DataAnalysisError("Failed to create rating distribution plot", str(e))

    def plot_genre_popularity(self, genre_data: Dict[str, Any]) -> str:
        """
        Create genre popularity charts

        Args:
            genre_data: Dictionary with genre analysis data

        Returns:
            Path to saved plot
        """
        try:
            logger.info("Creating genre popularity plot")

            if 'genres' not in genre_data or not genre_data['genres']:
                raise DataAnalysisError("Genre data must contain 'genres' list")

            # Convert to DataFrame
            genres_df = pd.DataFrame(genre_data['genres'])

            # Normalize column names to handle different naming conventions
            if 'avg_rating' in genres_df.columns and 'average_rating' not in genres_df.columns:
                genres_df['average_rating'] = genres_df['avg_rating']
            if 'popularity' in genres_df.columns and 'popularity_score' not in genres_df.columns:
                genres_df['popularity_score'] = genres_df['popularity']

            # Add default rating_count if not present
            if 'rating_count' not in genres_df.columns:
                genres_df['rating_count'] = genres_df.get('popularity_score', 0)

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Top genres by rating count
            top_genres = genres_df.nlargest(min(15, len(genres_df)), 'rating_count')
            ax1.barh(top_genres['genre'], top_genres['rating_count'], color='steelblue', edgecolor='black')
            ax1.set_xlabel('Number of Ratings', fontsize=11)
            ax1.set_title('Top 15 Genres by Rating Count', fontsize=13, fontweight='bold')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3, axis='x')

            # 2. Average rating by genre (top 15 by count)
            ax2.barh(top_genres['genre'], top_genres['average_rating'], color='coral', edgecolor='black')
            ax2.set_xlabel('Average Rating', fontsize=11)
            ax2.set_title('Average Rating by Genre (Top 15)', fontsize=13, fontweight='bold')
            ax2.set_xlim(0, 5)
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3, axis='x')

            # 3. Popularity score
            ax3.bar(range(len(top_genres)), top_genres['popularity_score'], color='lightgreen', edgecolor='black')
            ax3.set_xticks(range(len(top_genres)))
            ax3.set_xticklabels(top_genres['genre'], rotation=45, ha='right')
            ax3.set_ylabel('Popularity Score (%)', fontsize=11)
            ax3.set_title('Genre Popularity Score', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')

            # 4. Rating count vs Average rating scatter
            ax4.scatter(genres_df['average_rating'], genres_df['rating_count'],
                       s=100, alpha=0.6, c=genres_df['popularity_score'], cmap='viridis', edgecolors='black')
            ax4.set_xlabel('Average Rating', fontsize=11)
            ax4.set_ylabel('Rating Count', fontsize=11)
            ax4.set_title('Genre Rating vs Popularity', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(ax4.collections[0], ax=ax4)
            cbar.set_label('Popularity Score', fontsize=10)

            plt.tight_layout()

            # Generate unique filename with UUID
            unique_id = str(uuid.uuid4())[:8]
            filename = f"genre_popularity_{unique_id}.png"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Genre popularity plot saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create genre popularity plot: {str(e)}")
            raise DataAnalysisError("Failed to create genre popularity plot", str(e))

    def plot_time_series(self, time_data: Dict[str, Any]) -> str:
        """
        Create time-series visualization

        Args:
            time_data: Dictionary with time-series analysis data

        Returns:
            Path to saved plot
        """
        try:
            logger.info("Creating time-series plot")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Yearly trends
            if 'yearly_trends' in time_data and time_data['yearly_trends']:
                yearly_df = pd.DataFrame(time_data['yearly_trends'])

                # Plot rating count over years
                ax1_twin = ax1.twinx()
                line1 = ax1.plot(yearly_df['year'], yearly_df['rating_count'],
                                marker='o', linewidth=2, color='steelblue', label='Rating Count')
                line2 = ax1_twin.plot(yearly_df['year'], yearly_df['average_rating'],
                                     marker='s', linewidth=2, color='coral', label='Average Rating')

                ax1.set_xlabel('Year', fontsize=12)
                ax1.set_ylabel('Rating Count', fontsize=12, color='steelblue')
                ax1_twin.set_ylabel('Average Rating', fontsize=12, color='coral')
                ax1.set_title('Rating Trends Over Years', fontsize=14, fontweight='bold')
                ax1.tick_params(axis='y', labelcolor='steelblue')
                ax1_twin.tick_params(axis='y', labelcolor='coral')
                ax1.grid(True, alpha=0.3)

                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')

            # Day of week patterns
            if 'day_of_week_patterns' in time_data and time_data['day_of_week_patterns']:
                dow_df = pd.DataFrame(time_data['day_of_week_patterns'])

                # Create bar plot
                x_pos = range(len(dow_df))
                bars = ax2.bar(x_pos, dow_df['rating_count'], color='lightgreen', edgecolor='black', alpha=0.7)
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(dow_df['day'], rotation=45, ha='right')
                ax2.set_ylabel('Rating Count', fontsize=12)
                ax2.set_title('Rating Activity by Day of Week', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, dow_df['rating_count'])):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(value):,}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()

            # Generate unique filename with UUID
            unique_id = str(uuid.uuid4())[:8]
            filename = f"time_series_{unique_id}.png"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Time-series plot saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create time-series plot: {str(e)}")
            raise DataAnalysisError("Failed to create time-series plot", str(e))

    def generate_dashboard_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Create comprehensive HTML dashboard report

        Args:
            analysis_results: Dictionary with all analysis results

        Returns:
            Path to saved HTML report
        """
        try:
            logger.info("Generating dashboard report")

            # Generate unique filename with UUID
            unique_id = str(uuid.uuid4())[:8]
            filename = f"dashboard_report_{unique_id}.html"
            output_path = self.output_dir / filename

            # Generate HTML content
            html_content = self._build_html_dashboard(analysis_results)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Dashboard report saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate dashboard report: {str(e)}")
            raise DataAnalysisError("Failed to generate dashboard report", str(e))

    def _build_html_dashboard(self, data: Dict[str, Any]) -> str:
        """Build HTML dashboard content"""

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: #ffffff;
            padding: 20px;
            color: #24292f;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1280px;
            margin: 0 auto;
        }}
        header {{
        }}
        header h1 {{
            font-size: 32px;
            font-weight: 600;
            color: #24292f;
            margin-bottom: 8px;
        }}
        header p {{
            font-size: 14px;
            color: #57606a;
        }}
        .content {{
            padding: 0;
        }}
        .section {{
            margin-bottom: 48px;
        }}
        .section h2 {{
            color: #24292f;
            border-bottom: 1px solid #d0d7de;
            padding-bottom: 8px;
            margin-bottom: 16px;
            font-size: 24px;
            font-weight: 600;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 16px;
        }}
        .stat-card h3 {{
            font-size: 12px;
            color: #57606a;
            margin-bottom: 8px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: 600;
            color: #24292f;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            overflow: hidden;
        }}
        th {{
            background: #f6f8fa;
            color: #24292f;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            border-bottom: 1px solid #d0d7de;
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid #d0d7de;
            font-size: 14px;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background: #f6f8fa;
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
        <div class="content">
"""

        # Add time series summary first
        if 'time_series' in data and 'peak_activity' in data['time_series']:
            peak = data['time_series']['peak_activity']
            html += f"""
            <div class="section">
                <h2>Time Series Insights</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Most Active Year</h3>
                        <div class="value">{peak.get('most_active_year', 'N/A')}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Most Active Day</h3>
                        <div class="value">{peak.get('most_active_day', 'N/A')}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Highest Rated Year</h3>
                        <div class="value">{peak.get('highest_avg_rating_year', 'N/A')}</div>
                    </div>
                </div>
            </div>
"""

        # Add top movies section
        if 'top_movies' in data and data['top_movies']:
            html += """
            <div class="section">
                <h2>Top Rated Movies</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Title</th>
                            <th>Genres</th>
                            <th>Avg Rating</th>
                            <th>Rating Count</th>
                            <th>Weighted Score</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for i, movie in enumerate(data['top_movies'][:20], 1):
                html += f"""
                        <tr>
                            <td><strong>#{i}</strong></td>
                            <td>{movie.get('title', 'N/A')}</td>
                            <td>{movie.get('genres', 'N/A')}</td>
                            <td>{movie.get('average_rating', 0):.2f}</td>
                            <td>{movie.get('rating_count', 0):,}</td>
                            <td>{movie.get('weighted_rating', 0):.2f}</td>
                        </tr>
"""
            html += """
                    </tbody>
                </table>
            </div>
"""

        # Add genre analysis section
        if 'genre_analysis' in data and 'genres' in data['genre_analysis']:
            top_genres = data['genre_analysis']['genres'][:10]
            html += """
            <div class="section">
                <h2>Genre Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Genre</th>
                            <th>Avg Rating</th>
                            <th>Rating Count</th>
                            <th>Unique Movies</th>
                            <th>Popularity Score</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for genre in top_genres:
                html += f"""
                        <tr>
                            <td><strong>{genre.get('genre', 'N/A')}</strong></td>
                            <td>{genre.get('average_rating', 0):.2f}</td>
                            <td>{genre.get('rating_count', 0):,}</td>
                            <td>{genre.get('unique_movies', 0):,}</td>
                            <td>{genre.get('popularity_score', 0):.1f}%</td>
                        </tr>
"""
            html += """
                    </tbody>
                </table>
            </div>
"""

        html += """
        </div>

    </div>
</body>
</html>
"""
        return html
