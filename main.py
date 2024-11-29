import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the data
df = pd.read_csv('MoviesOnStreamingPlatforms.csv')

# Clean Rotten Tomatoes scores more carefully
def clean_rt_score(score):
    if pd.isna(score) or score == '':
        return np.nan
    return float(score.replace('/100', ''))

# Apply the cleaning function
df['Rotten_Tomatoes'] = df['Rotten Tomatoes'].apply(clean_rt_score)

# Convert age restrictions to numeric values (handling 'all' case)
age_mapping = {
    'all': 0,  # Setting 'all' to 0 years
    '7+': 7,
    '13+': 13,
    '16+': 16,
    '18+': 18
}
df['Age_Numeric'] = df['Age'].map(age_mapping)

# Remove rows with missing values for our analysis
df_clean = df.dropna(subset=['Age_Numeric', 'Rotten_Tomatoes'])

def descriptive_analysis():
    # Create DataFrame for each platform
    disney_movies = df_clean[df_clean['Disney+'] == 1]
    netflix_movies = df_clean[df_clean['Netflix'] == 1]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Age restrictions boxplot
    data_age = pd.DataFrame({
        'Disney+': disney_movies['Age_Numeric'],
        'Netflix': netflix_movies['Age_Numeric']
    })
    data_age.boxplot(ax=ax1)
    ax1.set_title('Age Restrictions Distribution')
    ax1.set_ylabel('Age Restriction')
    
    # Rotten Tomatoes boxplot
    data_rt = pd.DataFrame({
        'Disney+': disney_movies['Rotten_Tomatoes'],
        'Netflix': netflix_movies['Rotten_Tomatoes']
    })
    data_rt.boxplot(ax=ax2)
    ax2.set_title('Rotten Tomatoes Scores Distribution')
    ax2.set_ylabel('Score')
    
    plt.tight_layout()
    # Save the figure
    plt.savefig('descriptive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print basic information
    print("\nNumber of movies:")
    print(f"Disney+: {len(disney_movies)}")
    print(f"Netflix: {len(netflix_movies)}")
    
    # Print summary statistics
    print("\nAge Restriction Summary Statistics:")
    print("\nDisney+:")
    print(disney_movies['Age_Numeric'].describe())
    print("\nNetflix:")
    print(netflix_movies['Age_Numeric'].describe())
    
    print("\nRotten Tomatoes Score Summary Statistics:")
    print("\nDisney+:")
    print(disney_movies['Rotten_Tomatoes'].describe())
    print("\nNetflix:")
    print(netflix_movies['Rotten_Tomatoes'].describe())

def statistical_tests():
    disney_movies = df_clean[df_clean['Disney+'] == 1]
    netflix_movies = df_clean[df_clean['Netflix'] == 1]
    
    # Test for age restrictions (Mann-Whitney U test)
    age_stat, age_p = stats.mannwhitneyu(
        disney_movies['Age_Numeric'],
        netflix_movies['Age_Numeric'],
        alternative='less'
    )
    
    # Test for Rotten Tomatoes scores (Independent t-test)
    rt_stat, rt_p = stats.ttest_ind(
        disney_movies['Rotten_Tomatoes'],
        netflix_movies['Rotten_Tomatoes']
    )
    
    print("\nStatistical Test Results:")
    print(f"Age Restriction Test p-value: {age_p:.4f}")
    print(f"Rotten Tomatoes Score Test p-value: {rt_p:.4f}")
    print("\nInterpretation:")
    print("Age Restriction Test: H0: Disney+ ages are not lower than Netflix")
    print("Rotten Tomatoes Test: H0: There is no difference in mean scores")
    if age_p < 0.05:
        print("Age Test Result: Reject H0 - Disney+ has significantly lower age restrictions")
    else:
        print("Age Test Result: Failed to reject H0 - No evidence of lower age restrictions on Disney+")
    
    if rt_p < 0.05:
        print("RT Test Result: Reject H0 - There is a significant difference in scores")
    else:
        print("RT Test Result: Failed to reject H0 - No evidence of difference in scores")

def create_age_visualizations():
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Box Plot for Age Restrictions
    disney_ages = df_clean[df_clean['Disney+'] == 1]['Age_Numeric']
    netflix_ages = df_clean[df_clean['Netflix'] == 1]['Age_Numeric']
    
    ax1.boxplot([disney_ages, netflix_ages], labels=['Disney+', 'Netflix'])
    ax1.set_title('Age Restrictions Distribution')
    ax1.set_ylabel('Age Restriction (years)')
    
    # 2. Stacked Bar Chart for Age Categories
    platforms = ['Disney+', 'Netflix']
    age_categories = ['all', '7+', '13+', '16+', '18+']
    
    disney_dist = [(df_clean[(df_clean['Disney+'] == 1) & (df_clean['Age'] == cat)].shape[0] / 
                   df_clean[df_clean['Disney+'] == 1].shape[0]) * 100 for cat in age_categories]
    netflix_dist = [(df_clean[(df_clean['Netflix'] == 1) & (df_clean['Age'] == cat)].shape[0] / 
                    df_clean[df_clean['Netflix'] == 1].shape[0]) * 100 for cat in age_categories]
    
    x = np.arange(len(platforms))
    bottom = np.zeros(2)
    
    colors = ['lightgreen', 'yellowgreen', 'orange', 'salmon', 'red']
    for i, age in enumerate(age_categories):
        values = [disney_dist[i], netflix_dist[i]]
        ax2.bar(x, values, bottom=bottom, label=age, color=colors[i])
        bottom += values
    
    ax2.set_title('Age Rating Distribution (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(platforms)
    ax2.set_ylabel('Percentage')
    ax2.legend(title='Age Rating')
    
    plt.tight_layout()
    # Save the figure
    plt.savefig('age_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_boxplot():
    plt.figure(figsize=(10, 6))
    
    data_rt = pd.DataFrame({
        'Disney+': df_clean[df_clean['Disney+'] == 1]['Rotten_Tomatoes'],
        'Netflix': df_clean[df_clean['Netflix'] == 1]['Rotten_Tomatoes']
    })
    
    sns.boxplot(data=data_rt)
    plt.title('Rotten Tomatoes Score Distribution')
    plt.ylabel('Score')
    plt.savefig('quality_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_violin():
    plt.figure(figsize=(10, 6))
    
    data_rt = pd.DataFrame({
        'Disney+': df_clean[df_clean['Disney+'] == 1]['Rotten_Tomatoes'],
        'Netflix': df_clean[df_clean['Netflix'] == 1]['Rotten_Tomatoes']
    })
    
    sns.violinplot(data=data_rt)
    plt.title('Rotten Tomatoes Score Distribution (Violin Plot)')
    plt.ylabel('Score')
    plt.savefig('quality_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_histogram():
    plt.figure(figsize=(10, 6))
    
    disney_scores = df_clean[df_clean['Disney+'] == 1]['Rotten_Tomatoes']
    netflix_scores = df_clean[df_clean['Netflix'] == 1]['Rotten_Tomatoes']
    
    plt.hist(disney_scores, alpha=0.5, label='Disney+', bins=20, color='blue')
    plt.hist(netflix_scores, alpha=0.5, label='Netflix', bins=20, color='purple')
    plt.title('Distribution of Rotten Tomatoes Scores')
    plt.xlabel('Score')
    plt.ylabel('Number of Movies')
    plt.legend()
    plt.savefig('quality_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_kde():
    plt.figure(figsize=(10, 6))
    
    disney_scores = df_clean[df_clean['Disney+'] == 1]['Rotten_Tomatoes']
    netflix_scores = df_clean[df_clean['Netflix'] == 1]['Rotten_Tomatoes']
    
    sns.kdeplot(data=disney_scores, label='Disney+', color='blue')
    sns.kdeplot(data=netflix_scores, label='Netflix', color='red')
    plt.title('Density Distribution of Rotten Tomatoes Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('quality_kde.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_categories():
    plt.figure(figsize=(12, 6))
    
    def get_quality_category(score):
        if score >= 90: return 'Excellent (90-100)'
        elif score >= 80: return 'Very Good (80-89)'
        elif score >= 70: return 'Good (70-79)'
        elif score >= 60: return 'Fair (60-69)'
        else: return 'Poor (<60)'
    
    disney_movies = df_clean[df_clean['Disney+'] == 1]
    netflix_movies = df_clean[df_clean['Netflix'] == 1]
    
    disney_movies['Quality'] = disney_movies['Rotten_Tomatoes'].apply(get_quality_category)
    netflix_movies['Quality'] = netflix_movies['Rotten_Tomatoes'].apply(get_quality_category)
    
    categories = ['Poor (<60)', 'Fair (60-69)', 'Good (70-79)', 
                 'Very Good (80-89)', 'Excellent (90-100)']
    
    disney_dist = [len(disney_movies[disney_movies['Quality'] == cat]) / len(disney_movies) * 100 
                  for cat in categories]
    netflix_dist = [len(netflix_movies[netflix_movies['Quality'] == cat]) / len(netflix_movies) * 100 
                   for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, disney_dist, width, label='Disney+', color='blue', alpha=0.7)
    plt.bar(x + width/2, netflix_dist, width, label='Netflix', color='red', alpha=0.7)
    plt.title('Quality Score Distribution by Category')
    plt.xticks(x, categories, rotation=45)
    plt.ylabel('Percentage of Movies')
    plt.legend()
    plt.tight_layout()
    plt.savefig('quality_categories.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_time_series():
    plt.figure(figsize=(12, 6))
    
    years = sorted(df_clean['Year'].unique())
    disney_scores = [df_clean[(df_clean['Disney+'] == 1) & 
                            (df_clean['Year'] == year)]['Rotten_Tomatoes'].mean() 
                    for year in years]
    netflix_scores = [df_clean[(df_clean['Netflix'] == 1) & 
                             (df_clean['Year'] == year)]['Rotten_Tomatoes'].mean() 
                     for year in years]
    
    plt.plot(years, disney_scores, label='Disney+', color='blue', marker='o')
    plt.plot(years, netflix_scores, label='Netflix', color='red', marker='o')
    plt.title('Average Rotten Tomatoes Score by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('quality_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_summary_stats():
    disney_movies = df_clean[df_clean['Disney+'] == 1]
    netflix_movies = df_clean[df_clean['Netflix'] == 1]
    
    summary_stats = pd.DataFrame({
        'Disney+': disney_movies['Rotten_Tomatoes'].describe(),
        'Netflix': netflix_movies['Rotten_Tomatoes'].describe()
    })
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_stats.round(2).values,
                    rowLabels=summary_stats.index,
                    colLabels=summary_stats.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title('Rotten Tomatoes Score Summary Statistics', pad=20)
    plt.savefig('quality_summary_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_additional_visualizations():
    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Year Distribution Comparison (Histogram)
    disney_years = df_clean[df_clean['Disney+'] == 1]['Year']
    netflix_years = df_clean[df_clean['Netflix'] == 1]['Year']
    
    ax1.hist(disney_years, alpha=0.5, label='Disney+', bins=20, color='blue')
    ax1.hist(netflix_years, alpha=0.5, label='Netflix', bins=20, color='red')
    ax1.set_title('Movie Release Year Distribution')
    ax1.set_xlabel('Release Year')
    ax1.set_ylabel('Number of Movies')
    ax1.legend()
    
    # 2. Age vs Rotten Tomatoes Score (Scatter Plot)
    disney_data = df_clean[df_clean['Disney+'] == 1]
    netflix_data = df_clean[df_clean['Netflix'] == 1]
    
    ax2.scatter(disney_data['Age_Numeric'], disney_data['Rotten_Tomatoes'], 
               alpha=0.5, label='Disney+', color='blue')
    ax2.scatter(netflix_data['Age_Numeric'], netflix_data['Rotten_Tomatoes'], 
               alpha=0.5, label='Netflix', color='red')
    ax2.set_title('Age Restriction vs Rotten Tomatoes Score')
    ax2.set_xlabel('Age Restriction')
    ax2.set_ylabel('Rotten Tomatoes Score')
    ax2.legend()
    
    # 3. Quality Score Range Distribution
    def get_quality_range(score):
        if score >= 90: return 'Excellent (90-100)'
        elif score >= 80: return 'Very Good (80-89)'
        elif score >= 70: return 'Good (70-79)'
        elif score >= 60: return 'Fair (60-69)'
        else: return 'Poor (<60)'
    
    df_clean['Quality_Range'] = df_clean['Rotten_Tomatoes'].apply(get_quality_range)
    quality_ranges = ['Poor (<60)', 'Fair (60-69)', 'Good (70-79)', 
                     'Very Good (80-89)', 'Excellent (90-100)']
    
    disney_quality = df_clean[df_clean['Disney+'] == 1]['Quality_Range'].value_counts()
    netflix_quality = df_clean[df_clean['Netflix'] == 1]['Quality_Range'].value_counts()
    
    disney_percentages = [disney_quality.get(range_, 0) / len(disney_data) * 100 for range_ in quality_ranges]
    netflix_percentages = [netflix_quality.get(range_, 0) / len(netflix_data) * 100 for range_ in quality_ranges]
    
    x = np.arange(len(quality_ranges))
    width = 0.35
    
    ax3.bar(x - width/2, disney_percentages, width, label='Disney+', color='blue', alpha=0.7)
    ax3.bar(x + width/2, netflix_percentages, width, label='Netflix', color='red', alpha=0.7)
    ax3.set_title('Quality Score Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(quality_ranges, rotation=45)
    ax3.set_ylabel('Percentage of Movies')
    ax3.legend()
    
    # 4. Age Distribution Over Time
    years = sorted(df_clean['Year'].unique())
    disney_age_by_year = [df_clean[(df_clean['Disney+'] == 1) & 
                                  (df_clean['Year'] == year)]['Age_Numeric'].mean() 
                         for year in years]
    netflix_age_by_year = [df_clean[(df_clean['Netflix'] == 1) & 
                                   (df_clean['Year'] == year)]['Age_Numeric'].mean() 
                          for year in years]
    
    ax4.plot(years, disney_age_by_year, label='Disney+', color='blue', marker='o')
    ax4.plot(years, netflix_age_by_year, label='Netflix', color='red', marker='o')
    ax4.set_title('Average Age Restriction Over Time')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Average Age Restriction')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('additional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap():
    # Create correlation matrix for numerical columns
    numerical_cols = ['Year', 'Age_Numeric', 'Rotten_Tomatoes']
    correlation_matrix = df_clean[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_time_series_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group by year and platform, calculate mean scores
    yearly_scores = df_clean.groupby(['Year', 'Disney+'])['Rotten_Tomatoes'].mean().unstack()
    yearly_scores.plot(ax=ax1, marker='o')
    ax1.set_title('Average Rotten Tomatoes Score by Year')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Score')
    ax1.legend(['Netflix', 'Disney+'])
    
    # Count number of movies per year
    yearly_counts = df_clean.groupby(['Year', 'Disney+']).size().unstack()
    yearly_counts.plot(ax=ax2, marker='o')
    ax2.set_title('Number of Movies by Year')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Movies')
    ax2.legend(['Netflix', 'Disney+'])
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_age_analysis_plots():
    # Create figure with 2x3 subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    
    disney_movies = df_clean[df_clean['Disney+'] == 1]
    netflix_movies = df_clean[df_clean['Netflix'] == 1]
    
    # 1. Box Plot
    data_age = pd.DataFrame({
        'Disney+': disney_movies['Age_Numeric'],
        'Netflix': netflix_movies['Age_Numeric']
    })
    sns.boxplot(data=data_age, ax=ax1)
    ax1.set_title('Age Restrictions Distribution (Box Plot)')
    ax1.set_ylabel('Age Restriction')
    
    # 2. Violin Plot
    sns.violinplot(data=data_age, ax=ax2)
    ax2.set_title('Age Restrictions Distribution (Violin Plot)')
    ax2.set_ylabel('Age Restriction')
    
    # 3. Stacked Bar Chart for Age Categories
    platforms = ['Disney+', 'Netflix']
    age_categories = ['all', '7+', '13+', '16+', '18+']
    
    disney_dist = [(disney_movies['Age'] == cat).mean() * 100 for cat in age_categories]
    netflix_dist = [(netflix_movies['Age'] == cat).mean() * 100 for cat in age_categories]
    
    x = np.arange(len(platforms))
    bottom = np.zeros(2)
    
    colors = ['lightgreen', 'yellowgreen', 'orange', 'salmon', 'red']
    for i, age in enumerate(age_categories):
        values = [disney_dist[i], netflix_dist[i]]
        ax3.bar(x, values, bottom=bottom, label=age, color=colors[i])
        bottom += values
    
    ax3.set_title('Age Rating Distribution (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(platforms)
    ax3.set_ylabel('Percentage')
    ax3.legend(title='Age Rating')
    
    # 4. KDE Plot
    sns.kdeplot(data=disney_movies['Age_Numeric'], ax=ax4, label='Disney+', color='blue')
    sns.kdeplot(data=netflix_movies['Age_Numeric'], ax=ax4, label='Netflix', color='red')
    ax4.set_title('Age Distribution Density')
    ax4.set_xlabel('Age Restriction')
    ax4.set_ylabel('Density')
    ax4.legend()
    
    # 5. Histogram
    ax5.hist(disney_movies['Age_Numeric'], alpha=0.5, label='Disney+', bins=10, color='blue')
    ax5.hist(netflix_movies['Age_Numeric'], alpha=0.5, label='Netflix', bins=10, color='red')
    ax5.set_title('Age Distribution Histogram')
    ax5.set_xlabel('Age Restriction')
    ax5.set_ylabel('Count')
    ax5.legend()
    
    # 6. Age Distribution Over Time
    years = sorted(df_clean['Year'].unique())
    disney_age_by_year = [disney_movies[disney_movies['Year'] == year]['Age_Numeric'].mean() 
                         for year in years]
    netflix_age_by_year = [netflix_movies[netflix_movies['Year'] == year]['Age_Numeric'].mean() 
                          for year in years]
    
    ax6.plot(years, disney_age_by_year, label='Disney+', color='blue', marker='o')
    ax6.plot(years, netflix_age_by_year, label='Netflix', color='red', marker='o')
    ax6.set_title('Average Age Restriction Over Time')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Average Age Restriction')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('age_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_age_category_analysis():
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    disney_movies = df_clean[df_clean['Disney+'] == 1]
    netflix_movies = df_clean[df_clean['Netflix'] == 1]
    
    # 1. Pie Chart for Disney+
    disney_age_dist = disney_movies['Age'].value_counts()
    ax1.pie(disney_age_dist, labels=disney_age_dist.index, autopct='%1.1f%%',
            colors=['lightgreen', 'yellowgreen', 'orange', 'salmon', 'red'])
    ax1.set_title('Disney+ Age Distribution')
    
    # 2. Pie Chart for Netflix
    netflix_age_dist = netflix_movies['Age'].value_counts()
    ax2.pie(netflix_age_dist, labels=netflix_age_dist.index, autopct='%1.1f%%',
            colors=['lightgreen', 'yellowgreen', 'orange', 'salmon', 'red'])
    ax2.set_title('Netflix Age Distribution')
    
    plt.tight_layout()
    plt.savefig('age_distribution_pies.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_age_summary_stats():
    # Create summary statistics
    disney_movies = df_clean[df_clean['Disney+'] == 1]
    netflix_movies = df_clean[df_clean['Netflix'] == 1]
    
    summary_stats = pd.DataFrame({
        'Disney+': disney_movies['Age_Numeric'].describe(),
        'Netflix': netflix_movies['Age_Numeric'].describe()
    })
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_stats.round(2).values,
                    rowLabels=summary_stats.index,
                    colLabels=summary_stats.columns,
                    cellLoc='center',
                    loc='center')
    
    # Modify table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title('Age Restriction Summary Statistics', pad=20)
    plt.tight_layout()
    plt.savefig('age_summary_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run the analysis and save all plots
descriptive_analysis()
statistical_tests()
create_age_visualizations()
create_quality_boxplot()
create_quality_violin()
create_quality_histogram()
create_quality_kde()
create_quality_categories()
create_quality_time_series()
create_quality_summary_stats()
create_additional_visualizations()
create_heatmap()
create_time_series_analysis()
create_age_analysis_plots()
create_age_category_analysis()
create_age_summary_stats()