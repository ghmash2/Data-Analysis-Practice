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
    
    # Print basic information
    print("\nNumber of movies:")
    print(f"Disney+: {len(disney_movies)}")
    print(f"Netflix: {len(netflix_movies)}")
    
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
    plt.show()
    
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

# Run the analysis
descriptive_analysis()
statistical_tests()