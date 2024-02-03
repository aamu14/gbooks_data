import streamlit as st
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
from st_aggrid import AgGrid
import seaborn as sns

import pandas as pd
import numpy as np

# Configure
st.set_page_config(
    page_title="📚Google Book Dataset Analysis📚",
    layout="wide"
)
st.title('Googe Book Dataset Analysis')
st.sidebar.markdown('''
# Data Variables
title : the title of the book.\n
authors : name of the authors of the books (might include more than one author.\n
language : the language of the book \n
categories : the categories associated with the book (by Google store)\n
averageRating : the average rating of each book out of 5.\n
maturityRating : wheather the content of the book is for mature or NOT MATURE audience.\n
publisher : the name of the publisher.\n
publishedDate : when the book was published.\n
pageCount : number of pages of the books.\n
''')

st.markdown("""---""")
st.header("Data Preview")
url = "https://www.kaggle.com/datasets/bilalyussef/google-books-dataset"
st.markdown("""
            **The dataset used comes from Kaggle (%s).**""" % url)

import pandas as pd
import numpy as np

# Read the CSV file
my_data = pd.read_csv("google_books_dataset.csv")
st.markdown("""
            First, let's see how the dataset looks like.
""")
st.write(my_data.head())

st.markdown("""
            This table has 1025 rows and 9 columns. From the table above, it is evident that some columns contain missing values, different formats, and possibly dirty data. Next, I will perform data cleaning and manipulation to prepare the data for use.
""")
st.markdown("""---""")

st.header("Data Cleaning and Data Manipulation")
# Drop the first column
my_data = my_data.iloc[:, 1:]

st.markdown("""
            In the analysis of the table, columns with empty data will be converted to NA. Let's compare the data before and after cleaning to see the effect of the process.
""")

# Replace empty strings with NaN
my_data = my_data.replace('', np.nan)
# Extract text within brackets
my_data['extracted_authors'] = my_data['authors'].str.extract(r"\[(.*?)\]")
my_data['extracted_authors'].fillna("Not Available", inplace=True)
# Modify the extraction regex to exclude quotes
my_data['extracted_authors'] = my_data['authors'].str.extract(r"'(.*?)'")
my_data['extracted_authors'].fillna("Not Available", inplace=True)
my_data['extracted_authors'] = my_data['extracted_authors'].apply(str)

# Replace empty strings with NaN
my_data = my_data.replace('', np.nan)
missing_values_report_before = my_data.isnull().sum().to_frame(name='Missing Values')
missing_values_report_before['% Missing'] = (missing_values_report_before['Missing Values'] / len(my_data)) * 100

my_data['categories'] = my_data['categories'].str.extract(r"'([^']*)'")
def replace_authors(authors):
  if pd.isna(authors):
    return "Not Available"
  else:
    return "Available"

my_data['authors'] = my_data['authors'].apply(replace_authors)
my_data['categories'].fillna("Not Categorized", inplace=True)
my_data['averageRating'] = my_data['averageRating'].astype(str)
my_data['averageRating'].fillna("No Rating", inplace=True)
my_data['publisher'].fillna("No Identified", inplace=True)
my_data['publishedDate'].fillna("Unknown", inplace=True)
my_data['publishedDate'] = my_data['publishedDate'].astype(str)
my_data['publishedYear'] = my_data['publishedDate'].str[:4]  # Extract first 4 characters
my_data.loc[my_data['publishedDate'] == 'Unknown', 'publishedYear'] = 'Unknown'

# Create a copy of the DataFrame to avoid modifying the original
new_data = my_data.copy()

# Create the new column in the copied DataFrame
new_data['NewpageCount'] = pd.NA  # Initialize with NA

# Handle "Not Counted" values
new_data.loc[new_data['pageCount'] == 'Not Counted', 'NewpageCount'] = "Not Counted"  # Assign "Not Counted" category

# Classify known page counts
classification_ranges = [0, 100, 300, 500, 800, 1500, float('inf')]  # Adjust as needed
classification_labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Epic']
new_data.loc[new_data['pageCount'] != 'Not Counted', 'NewpageCount'] = pd.cut(new_data.loc[new_data['pageCount'] != 'Not Counted', 'pageCount'], bins=classification_ranges, labels=classification_labels)

# Convert NewpageCount to string
new_data['NewpageCount'] = new_data['NewpageCount'].astype(str)

# Replace 'nan' with 'Not Counted' in all string columns
new_data = new_data.replace('nan', 'Not Counted', regex=False)  # Use regex=False for exact matching
new_data['averageRating'] = new_data['averageRating'].replace('Not Counted', 'No Rating', regex=False)  # Use regex=False for exact matching

# Delete the 'pageCount' column from the copied DataFrame
del new_data['pageCount']
del new_data['publishedDate']
# The original DataFrame 'my_data' remains unchanged
# The modified data is now in the 'new_data' variable

#Check cleaned data
missing_values_report_after = new_data.isnull().sum().to_frame(name='Missing Values')
missing_values_report_after['% Missing'] = (missing_values_report_after['Missing Values'] / len(my_data)) * 100

col1, col2 = st.columns(2, gap="small")

with col1:
    st.subheader("Before cleaned")
    st.write(missing_values_report_before)
with col2:
    st.subheader("After cleaned")
    st.write(missing_values_report_after)
#Check Missing Value Before Cleaned

st.markdown("""
            It can be seen that there are several columns with missing data, including columns such as authors, categories, averageRating, publisher, publishedDate, and pageCount. i will do some data manipulatins. The steps to handle these data are as follows:

1. authors: Adjustment will be made based on the data. If the data contains the author's name(s), it will be labeled as 'Available'. If not, it will be labeled as 'Not Available'. \n
2. extracted_authors: This new column is created by extracting data from authors column using regex. The difference from the 'authors' column is that all entries in the 'authors' column will be labeled as either 'Available' or 'Not Available,' whereas the 'extracted_authors' column will retain the actual names of the authors for other purposes.
3. categories: Any missing values will be changed to 'Not Categorized'. \n
4. averageRating: Any missing values will be labeled as 'No Rating'. \n
5. publisher: Any missing values will be labeled as 'Not Identified'. \n
6. publishedDate: Any missing values will be labeled as 'Unknown' and this column will be extracted into year only. this column will be changed into 'publishedYear'. \n
7. pageCount: Any missing values will be labeled as 'Not Counted'. The result will be changed into page interval and make a new categories based on how many pages the books have. it will be saved into 'NewpageCount'. \n

Here is the data after being cleaned:
""")
# Define the desired column order
column_order = ['title','authors', 'extracted_authors', 'language', 'categories', 'NewpageCount', 'maturityRating', 'averageRating', 'publisher', 'publishedYear']

# Reorder the columns in the DataFrame
new_data = new_data[column_order]
st.dataframe(new_data)
st.markdown("Next, we can create basic visualizations (though not interactive) for all columns.")
st.markdown("""---""")
#batas

#batas
st.header("Visualization Graph")
# Secara Keseluruhan Penggunaan
ibook1 = st.selectbox('What do you want to see?',
                               ['authors', 'extracted_authors', 'language', 'categories', 'page categories', 'maturityRating', 'averageRating', 'publisher', 'publishedYear']
, key="book 1")

if ibook1 == 'authors':
    # Count occurrences including null values
    authors_counts_data = new_data['authors'].value_counts(dropna=False)
    # Exclude "Not Available" and sum remaining counts  
    have_authors_count = authors_counts_data.drop('Not Available').sum()

    # Get count of "Not Available" directly
    not_available_count = authors_counts_data['Not Available']

    # Set labels and data
    labels = ['Available', 'Not Available']
    counts = [have_authors_count, not_available_count]

    # Calculate percentages
    total_books = sum(counts)
    percentages = [count / total_books * 100 for count in counts]
    colors = ['green', 'red']
    # Create a custom label for each wedge
    label_values = [f"{label}\n{count} ({percentage:.1f}%)" for label, count, percentage in zip(labels, counts, percentages)]

    # Create the pie chart
    author_pie = plt.figure(figsize=(15, 7)) # Adjust figure size as needed
    plt.pie(counts, labels=label_values, autopct="", colors=colors)

    # Customize pie chart elements
    plt.title('Distribution of Books by Authors')
    plt.axis('equal')  # Ensure a circular pie chart    
    # Automatically adjust layout to prevent overlap
    plt.tight_layout()
    plt.axis('equal')  # Ensure a circular pie chart

    # Show the graph with a custom figure size
    st.pyplot(author_pie, clear_figure=True)
    st.markdown("""
    Take a look at the numbers — 767 books have names of authors listed, but 258 books don't. There could be a few reasons behind this. Some older books might not have complete author details recorded. Also, it's possible that mistakes were made when recording information, like forgetting to note down the author's name. It's interesting to see how factors like the age of the books and human errors can play a role in whether or not we have information about the authors.
    """)
elif ibook1 == 'extracted_authors':
    top_5_authors = new_data['extracted_authors'].value_counts(dropna=False).head(6).drop('Not Available')

    # Extract data for the plot
    authors = top_5_authors.index.to_list()  # Get author names
    counts = top_5_authors.to_list()  # Get corresponding counts

    # Sort data for the horizontal bar chart
    sorted_indices = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
    authors = [authors[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    # Reverse the order to have the largest on top
    authors.reverse()
    counts.reverse()

    # Create the horizontal bar chart
    extracted_authors_barh= plt.figure(figsize=(15, 7))  # Adjust figure size as needed
    bars = plt.barh(authors, counts, color='lightcoral')  # Use barh for horizontal bars

    # Add annotations next to bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())),
             va='center', ha='left', fontsize=10)

    # Customize plot elements
    plt.xlabel('Frequency')
    plt.ylabel('Authors')
    plt.title('Top 5 Authors with the Most Book Publications')
    plt.tight_layout()
    
    # Show the plot
    st.pyplot(extracted_authors_barh)

    st.markdown("""
    
    In the provided data, Wendelin Van Draanen emerges as the most productive author, with an impressive total of 51 published books. Jessica Keyes follows closely with a notable collection of 19 books. Casenote L.B. holds the third position with 15 books, followed by Marian K. with 9 books, and Casenotes with 6 books. This distribution highlights the diverse contributions of these authors, showcasing Van Draanen's significant body of work and the varying levels of output among the other authors in the list.
    """)
elif ibook1 == 'language':
    # Get the language counts
    language_counts = new_data['language'].value_counts()

    # Create the bar chart
    languange_bar= plt.figure(figsize=(15, 7))  # Adjust figure size as needed

    # Store the bar objects in a variable (crucial for annotation)
    bars = plt.bar(language_counts.index, language_counts.values)
    # Calculate maximum annotation height for spacing
    max_height = max(bar.get_height() for bar in bars)

    # Add annotations above bars, extending the vertical interval
    for bar in bars:
        height = bar.get_height()
        y_pos = min(height , 1 * max_height)  # Expanded vertical offset
        plt.annotate(f'{height}',
                                 xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                                 ha='center',
                                 va='bottom',
                                 fontsize=10
        )
    # Customize chart elements
    plt.xlabel('Language')
    plt.ylabel('Frequency')
    plt.title('Distribution of Book by Languages')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability

    # Show the plot
    plt.tight_layout()  # Adjust spacing for better visibility
    st.pyplot(languange_bar)

    st.markdown("""
    In the 'Language' column, ISO 639-1 language codes are used to show the language names. The majority of books, around 966, are in English ('en'). Following closely is Arabic ('ar') with 57 books. Additionally, there's a small but notable representation of German ('de') and Swedish ('sv'), each with 1 book. This breakdown gives us a glimpse into the variety of languages found in our dataset.    """)

elif ibook1 == 'categories':
    top_5_categories = new_data['categories'].value_counts(dropna=False).head(6).drop('Not Categorized')

    # Extract data for the plot
    categories = top_5_categories.index.to_list()  # Get category names
    counts = top_5_categories.to_list()  # Get corresponding counts

    # Sort data for the horizontal bar chart
    sorted_indices = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
    categories = [categories[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    # Reverse the order to have the largest on top
    categories.reverse()
    counts.reverse()

    # Create the horizontal bar chart
    categories_bar= plt.figure(figsize=(15, 7))  # Adjust figure size as needed
    bars = plt.barh(categories, counts, color='skyblue')  # Use barh for horizontal bars

    # Add annotations next to bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())),
                 va='center', ha='left', fontsize=10)

    # Customize plot elements
    plt.xlabel('Frequency')
    plt.ylabel('Category')
    plt.title('Top 5 Book Categories')
    plt.tight_layout()

    # Show the plot
    st.pyplot(categories_bar)
    st.markdown("""
    As you can see, the 'Computers' category has the most books in our dataset, totaling 90. Following closely are 'Juvenile Fiction' with 61 books, 'Fiction' with 41 books, 'History' with 38 books, and 'Business & Economics' with 32 books. It's important to notice that even though we have 1025 rows, no single category has over 100 books. This suggests a wide range of categories, each contributing to the total of 1025 books in our dataset. \n
    Now, let's see if there is a book that does not have any category:
    """)

    category_counts = new_data['categories'].value_counts(dropna=False)

    have_category_count = category_counts.drop('Not Categorized').sum()
    no_category_count = category_counts['Not Categorized']

    # Create the plot
    labels = ['Have Category', 'Not Categorized']
    counts = [have_category_count, no_category_count]

    # Calculate percentage
    percentage_have_category = (have_category_count / (have_category_count + no_category_count)) * 100
    percentage_no_category = (no_category_count / (have_category_count + no_category_count)) * 100

    # Create the pie chart
    pie_category = plt.figure(figsize=(15, 7))  # Adjust figure size as needed
    plt.pie(counts, labels=['Have Category\n{:.1f}% ({})'.format(percentage_have_category, have_category_count),
                        'Not Categorized\n{:.1f}% ({})'.format(percentage_no_category, no_category_count)],
        autopct="", colors=['skyblue', 'lightcoral'])
    # Customize pie chart elements
    plt.title('Distribution of Books by Category')
    plt.axis('equal')  # Ensure a circular pie chart

    # Optionally explode the "Have Category" slice for emphasis
    st.pyplot(pie_category)
    st.markdown("""It appears that there are 120 books did not recorded categories. Let's investigate the publication years of these uncategorized books.""")

    # Filter books with the category 'Not Categorized'
    not_categorized_books = new_data[new_data['categories'] == 'Not Categorized']

    # Group by publishedYear and count the number of 'Not Categorized' books for each year
    not_categorized_counts = not_categorized_books.groupby('publishedYear').size()

    #  Create a bar chart
    book_uncategory_year= plt.figure(figsize=(15, 7))
    not_categorized_counts.plot(kind='bar', color='lightcoral')
    plt.title('Number of Books with "Not Categorized" by Published Year')
    plt.xlabel('Published Year')
    plt.ylabel('Number of Books')
    st.pyplot(book_uncategory_year)
    st.markdown("""It indicates that both older and more recent books lack assigned categories. The absence of categories in older books might be intentional, but for recent publications, the absence could potentially came from human error, leading to unrecorded categories. Notably, the maximum count of uncategorized books corresponds to those without a specified publication year. This may suggest these books are truly aged, possibly damaged, or simply not documented.""")

elif ibook1 == 'page categories':
    st.markdown("""
    For page of books, i split them into several book categories: \n
    1. 0-100 pages → Very Short \n
    2. 100-300 pages → Short \n
    3. 300-500 pages → Medium \n
    4. 500-800 pages → Long \n
    5. 800-1500 pages → Very Long \n
    6. more than 1500 pages → Epic \n
    7. Not Counted
""")

    # Get the category counts
    page_category_counts = new_data['NewpageCount'].value_counts()

    # Create the x and y axes labels
    page_categories = page_category_counts.index.to_list()
    page_counts = page_category_counts.to_list()

    # Create the bar plot
    page_bar= plt.figure(figsize=(15, 7))  # Adjust figure size if needed
    bar_colors = plt.cm.viridis(range(len(page_categories)))  # Set custom color scheme (optional)
    bars = plt.bar(page_categories, page_counts, color=bar_colors)

    # Calculate maximum annotation height for spacing
    max_height = max(bar.get_height() for bar in bars)

    # Add annotations above bars, extending the vertical interval
    for bar in bars:
        height = bar.get_height()
        y_pos = min(height , 1 * max_height)  # Expanded vertical offset
        plt.annotate(f'{height}',
                      xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                      ha='center',
                      va='bottom',
                    fontsize=10
                    )
    # Add labels and title
    plt.xlabel('Categories')
    plt.ylabel('Page Count')
    plt.title('Distribution of Books by Page Count')

    # Customize tick labels and rotation
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    st.pyplot(page_bar)
    st.markdown("""
    We can observe that books with fewer pages are more abundant. However, there is an exception for the 'Very Short' category. Additionally, books with uncounted pages are the most numerous. We might want to explore the distribution with the publication years or publishers. \n
    """)
    #Comparison

    # Filter books with 'Not Counted' in 'NewpageCount'
    not_counted_books = new_data[new_data['NewpageCount'] == 'Not Counted']

    # Group by publishedYear and count the number of 'Not Counted' books for each year
    not_counted_counts = not_counted_books.groupby('publishedYear').size()

    # Select the top 10 years and sort by values
    top_10_years = not_counted_counts.nlargest(10)

    # Create a horizontal bar chart for the top 10 years (sorted)
    page_year_bar = plt.figure(figsize=(15, 7))
    top_10_years.sort_values().plot(kind='barh', color='lightcoral')  # Sort values before plotting
    plt.title('Top 10 Published Years with Most Books "Not Counted" in NewpageCount')
    plt.xlabel('Number of Books')
    plt.ylabel('Published Year')

    # Display labels for each bar
    for index, value in enumerate(top_10_years.sort_values()):
        plt.text(value + 0.1, index, str(value), va='center', ha='left', fontsize=10)

    # Adjust font size of y-axis labels
    plt.yticks(fontsize=10)
    #batas
    # Filter books with 'Not Counted' in 'NewpageCount'
    not_counted_books = new_data[new_data['NewpageCount'] == 'Not Counted']

    # Group by publisher and count the number of 'Not Counted' books for each publisher
    not_counted_counts_by_publisher = not_counted_books.groupby('publisher').size()

    # Select the top 10 publishers
    top_10_publishers_page = not_counted_counts_by_publisher.nlargest(10)

    # Create a horizontal bar chart for the top 10 publishers
    top10_page_uncounted_publisher = plt.figure(figsize=(15, 7))
    top_10_publishers_page.sort_values().plot(kind='barh', color='lightcoral')  # Sort values before plotting
    plt.title('Top 10 Publishers with Most Books "Not Counted" in NewpageCount')
    plt.xlabel('Number of Books with Uncounted Page')
    plt.ylabel('Publisher')

    # Display labels for each bar
    for index, value in enumerate(top_10_publishers_page.sort_values()):
        plt.text(value + 0.1, index, str(value), va='center', ha='left', fontsize=10)

    # Adjust font size of y-axis labels
    plt.yticks(fontsize=10)

    col3, col4 = st.columns(2, gap="small")
    with col3:
        st.subheader("Distribution by Published Year")
        st.pyplot(page_year_bar)
    with col4:
        st.subheader("Distribution by Publisher")
        st.pyplot(top10_page_uncounted_publisher)
    st.markdown("""
Both graphs show that books with uncounted pages often have missing information, such as unknown publication years and unidentified publishers. Although the maximum value may not be high, it reveals the diversity within the data, encompassing a wide range of years and various publishers."
                """)


elif ibook1 == 'maturityRating':

    # Count occurrences including null values
    maturity_rating_counts_data = new_data['maturityRating'].value_counts(dropna=False)

    # Exclude "NOT_MATURE" and sum remaining counts
    have_maturity_rating_count = maturity_rating_counts_data.drop('NOT_MATURE').sum()

    # Get count of "NOT_MATURE" directly
    not_available_count_maturity = maturity_rating_counts_data['NOT_MATURE']

    # Set labels and data
    labels_maturity = ['MATURE', 'NOT_MATURE']
    counts_maturity = [have_maturity_rating_count, not_available_count_maturity]

    # Calculate percentages
    total_books_maturity = sum(counts_maturity)
    percentages_maturity = [count / total_books_maturity * 100 for count in counts_maturity]
    colors_maturity = ['red', 'green']

    # Create a custom label for each wedge
    label_values_maturity = [f"{label}\n{count} ({percentage:.1f}%)" for label, count, percentage in zip(labels_maturity, counts_maturity, percentages_maturity)]

    # Create the pie chart
    maturity_pie = plt.figure(figsize=(15, 7))  # Adjust figure size as needed
    plt.pie(counts_maturity, labels=label_values_maturity, autopct="", colors=colors_maturity)

    # Customize pie chart elements
    plt.title('Distribution of Books by Maturity Rating')
    plt.axis('equal')  # Ensure a circular pie chart

    # Automatically adjust layout to prevent overlap
    plt.tight_layout()
    st.pyplot(maturity_pie)
    st.markdown("""
    It's noticeable that the majority of the data is labeled as 'Not Mature.' It seems unusual that books categorized under subjects like computers and dissertations would most likely have a low likelihood of being labeled as 'Not Mature.' Without additional information, it's challenging to draw any conclusions. However, we can proceed with other analyses.
    """)

elif ibook1 == 'averageRating':
    # Count occurrences including null values
    average_rating_counts_data = new_data['averageRating'].value_counts(dropna=False)

    # Exclude "Not Counted" and sum remaining counts
    have_average_rating_count = average_rating_counts_data.drop('No Rating').sum()

    # Get count of "Not Counted" directly
    not_counted_count_average_rating = average_rating_counts_data['No Rating']

    # Set labels and data
    labels_average_rating = ['Has Rating', 'No Rating']
    counts_average_rating = [have_average_rating_count, not_counted_count_average_rating]

    # Calculate percentages
    total_books_average_rating = sum(counts_average_rating)
    percentages_average_rating = [count / total_books_average_rating * 100 for count in counts_average_rating]
    colors_average_rating = ['green', 'red']

    # Create a custom label for each wedge
    label_values_average_rating = [f"{label}\n{count} ({percentage:.1f}%)" for label, count, percentage in zip(labels_average_rating, counts_average_rating, percentages_average_rating)]

    # Create the pie chart
    average_rating_pie = plt.figure(figsize=(15, 7))  # Adjust figure size as needed
    plt.pie(counts_average_rating, labels=label_values_average_rating, autopct="", colors=colors_average_rating)

    # Customize pie chart elements
    plt.title('Distribution of Books by Average Rating')    
    plt.axis('equal')  # Ensure a circular pie chart

    # Automatically adjust layout to prevent overlap
    plt.tight_layout()
   

    # Separate known and unknown ratings
    unknown_rating_df = new_data[new_data['averageRating'] == 'No Rating']
    known_rating_df = new_data[new_data['averageRating'] != 'No Rating']

    # Count occurrences for known and unknown ratings
    known_counts = known_rating_df['averageRating'].value_counts()
    unknown_count = unknown_rating_df['averageRating'].value_counts().get('No Rating', 0)
    
    # Combine known counts and unknown count
    combined_counts = known_counts._append(pd.Series({'No Rating': unknown_count}))

    # Create the bar plot
    average_bar= plt.figure(figsize=(15, 7))
    combined_counts.sort_index().plot(kind='bar', color='skyblue')

    # Add annotations above bars
    for index, value in enumerate(combined_counts.sort_index()):
        plt.text(index, value + 0.1, str(value), ha='center', va='bottom', fontsize=10)

    # Customize plot elements
    plt.xlabel('Average Rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of Books by Average Rating')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability

    # Show the plot
    plt.tight_layout()

    col5, col6 = st.columns(2, gap="small")
    with col5:
        st.subheader("Percentage of Average Rating")
        st.pyplot(average_rating_pie)
    with col6:
        st.subheader("Distribution of Average Rating")
        st.pyplot(average_bar)
   
    st.markdown("""
    The pie chart shows that most books, around 83.7% (858), don't have any ratings, while the remaining 16.3% (167) do. The bar chart provides a closer look, displaying the number of rated books, from 'No Rating' to ratings between 1.0 and 5.0. Notably, there are no books with an average rating of 1.5. To dig deeper, we can explore books without ratings and how they're spread across different publication years.
    """)

    # Assuming 'new_data' is your DataFrame
    no_rating_year_bar = plt.figure(figsize=(15, 7))

    # Filter out books with 'No Rating'
    no_rating_books = new_data[new_data['averageRating'] == 'No Rating']

    # Group by publishedYear and count the number of 'No Rating' books for each year
    no_rating_counts_by_year = no_rating_books.groupby('publishedYear').size()

    # Plot the bar chart
    no_rating_counts_by_year.plot(kind='bar', color='lightcoral')

    # Add labels to each bar (only when the value is >= 5)
    for index, value in enumerate(no_rating_counts_by_year):
        if value >= 10:
            plt.text(index, value + 1, str(value), ha='center', va='bottom', fontsize=10)

    # Add a line following the top of the bars
    plt.plot(no_rating_counts_by_year.index, no_rating_counts_by_year.values, color='blue', linestyle='solid', linewidth=2, markersize=5, alpha= 0.5)

    # Customize plot elements
    plt.title('Distribution of Books with "No Rating" Based on Published Year')
    plt.xlabel('Published Year')
    plt.ylabel('Number of Books')

    # Show the plot
    plt.tight_layout()
    st.pyplot(no_rating_year_bar)
    
    st.markdown("""
    In this graph, I display labels only for years with 10 or more books marked as "No Rating." The line aids in comprehending the annual trend. It is evident that the count increases as we approach 2019. One plausible hypothesis is that newer book recordings have a higher incidence of "No Rating," contributing to a continual rise each year. However, there is an exception for the unknown years, suggesting the possibility of human error during data recording.
                """)

elif ibook1 == 'publisher':
    # Count occurrences including null values
    publisher_counts_data = my_data['publisher'].value_counts(dropna=False)

    # Exclude "No Identified" and sum remaining counts
    have_publisher_count = publisher_counts_data.drop('No Identified').sum()

    # Get count of "No Identified" directly
    no_identified_count = publisher_counts_data['No Identified']

    # Set labels and data
    labels = ['Have Publisher', 'No Publisher']
    counts = [have_publisher_count, no_identified_count]

    # Calculate percentages
    total_books = sum(counts)
    percentages = [count / total_books * 100 for count in counts]

    # Create the pie chart with labels
    pie_publisher=plt.figure(figsize=(15, 7))  # Adjust figure size as needed
    plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=['skyblue', 'lightcoral'])

    # Customize pie chart elements
    plt.title('Distribution of Books by Publisher')
    plt.axis('equal')  # Ensure a circular pie chart

    # Optionally explode the "Have Publisher" slice for emphasis
    st.pyplot(pie_publisher)
    st.markdown("""
    We observe that the number of books with identified publishers is slightly higher than those without publishers. However, it raises concerns as books without a publisher account for 43.7% of the total books. Further investigation might needed (indicating whether the book is old or not) or if it's simply due to human error resulting in the absence of recorded publisher information.            
    """)

    # Get the top 10 publishers (excluding "No Identified") and their counts
    top_5_publishers = my_data['publisher'].value_counts().head(6)  # Get 11 to ensure 10 after exclusion
    top_5_publishers = top_5_publishers.drop('No Identified')  # Exclude "No Identified"

    # Extract data for the plot
    publishers = top_5_publishers.index.to_list()
    counts = top_5_publishers.to_list()

    # Create the bar plot
    top5_publisher= plt.figure(figsize=(15, 7))  # Adjust figure size as needed


    # Store the bar objects in a variable (crucial for annotation)
    bars = plt.bar(publishers, counts, color='skyblue')
    # Calculate maximum annotation height for spacing
    max_height = max(bar.get_height() for bar in bars)

    # Add annotations above bars, extending the vertical interval
    for bar in bars:
        height = bar.get_height()
        y_pos = min(height , 1 * max_height)  # Expanded vertical offset
        plt.annotate(f'{height}',
                      xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                      ha='center',
                      va='bottom',
                    fontsize=10
                    )

    # Customize plot elements
    plt.xlabel('Publisher')
    plt.ylabel('Number of Books')
    plt.title('Top 5 Publishers (Excluding "No Identified")')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Show the plot
    plt.tight_layout()
    st.pyplot(top5_publisher)
    st.markdown("""
    This represents the top 5 publishers and the number of books they have published. Yearling emerges as the most frequent publisher with 37 books, followed by Knopf Books with 21 books, CRC Press with 20 books, J. Wiley & Sons with 18 books, and Routledge with 16 books. Similar to other columns, the maximum value is not significantly high, indicating a diverse range of publishers in the dataset. Additionally, this count is influenced by books with no specified publisher, which accounts for 43.7% of the total books.
    """)

else:
    st.markdown("""
In this column, I processed and categorized the data based on published years, grouping them into intervals of 100 years. The 'Unknown' years are presented separately but in the same graph for clarity, as illustrated in the graph below:    
    """)

    problematic_indices = my_data.index[my_data['publishedYear'] == '195?']
    my_data.loc[problematic_indices, 'publishedYear'] = 'Unknown'
    known_years_df_real = my_data[my_data['publishedYear'] != 'Unknown']
    known_years_df= known_years_df_real['publishedYear']
    try:
        known_years_df = pd.to_numeric(known_years_df).astype('int64')  # Directly convert the Series to numeric
    except ValueError:
        print("Error converting 'known_years_df' to integer. Check for remaining non-numeric values.")
    known_years_df = pd.cut(known_years_df, bins=[1600, 1700, 1800, 1900, 2000, 2020])

    unknown_years_df = my_data[my_data['publishedYear'] == 'Unknown']
    unknown_years_df= unknown_years_df['publishedYear']
    unknown_years_df.value_counts()

    # Extract data for known and unknown years
    known_counts = known_years_df.value_counts()
    unknown_count = unknown_years_df.value_counts().to_list()[0]

    # Define labels and counts for combined plot
    labels_combined = ['(2000, 2020]', '(1900, 2000]', '(1800, 1900]', '(1700, 1800]', '(1600, 1700]', 'Unknown']
    counts_combined = known_counts.to_list() + [unknown_count]

    # Create the combined bar plot
    bar_year=plt.figure(figsize=(15, 7))  # Adjust figure size as needed


    # Customize plot elements
    plt.xlabel('Published Year or Decade')
    plt.ylabel('Number of Books')
    plt.title('Distribution of Books by Published Year (Known and Unknown)')
    plt.xticks(rotation=45, ha='right')

    # Store the bar objects in a variable (crucial for annotation)
    bars = plt.bar(labels_combined, counts_combined, color=['skyblue', 'skyblue', 'skyblue', 'skyblue', 'skyblue', 'lightcoral'])
    # Calculate maximum annotation height for spacing
    max_height = max(bar.get_height() for bar in bars)

    # Add annotations above bars, extending the vertical interval
    for bar in bars:
        height = bar.get_height()
        y_pos = min(height , 1 * max_height)  # Expanded vertical offset
        plt.annotate(f'{height}',
                      xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                      ha='center',
                      va='bottom',
                      fontsize=10
                    )
    # Show the plot
    plt.tight_layout()
    st.pyplot(bar_year)

    st.markdown("""
    as you can see, there are so many books published started from 1900 until 2019, like in year 1900-2000 there are 406 books published, and in year 2000-2019 there are 531 books published.  It makes sense that older books have fewer records, possibly due to challenges in documentation, limited information available, damaged books, and various other reasons. \n
    Next, let's examine the trend of book publications each year.
    """)

    known_years_df_real = my_data[my_data['publishedYear'] != 'Unknown']
    # Count books published each year
    yearly_counts = known_years_df_real['publishedYear'].value_counts().sort_index()  # Sort chronologically

    # Create the line plot
    books_trend= plt.figure(figsize=(15, 7))
    plt.plot(yearly_counts.index, yearly_counts.values, color='skyblue')

    # Set x-axis limits to match data range
    plt.xlim(yearly_counts.index.min(), yearly_counts.index.max())

    # Add text labels for minimum and maximum years
    plt.xticks(ticks=[yearly_counts.index.min(), yearly_counts.index.max()],
               labels=[str(yearly_counts.index.min()), str(yearly_counts.index.max())])
    # Obtain first and last indices
    first_index = yearly_counts.index[0]
    last_index = yearly_counts.index[-1]

    # Set x-axis limits and ticks (corrected to show only first and last labels)
    plt.xlim(first_index, last_index)  # Set limits to match the first and last indices
    plt.xticks(ticks=[first_index, last_index],labels=[str(first_index), str(last_index)])  # Removed


    # Customize plot elements
    plt.xlabel('Published Year')
    plt.ylabel('Number of Books')
    plt.title('Number of Books Published Each Year')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    # Hide x-axis text # Pass an empty list to hide all text labels

    plt.tight_layout()
    st.pyplot(books_trend)
    st.markdown("""
    With this graph, we can clearly see the trend, demonstrating an increasing exponentially in book publications each year. It also indicates that the dataset dominantly consists of newer books rather than older ones.
   """)
st.markdown("""---""")
st.header("""Conclusion""")
st.markdown("""
1. The dataset initially comprises 1025 rows and 9 columns. However, further data manipulation and cleaning will be necessary due to missing values and varying formats in certain columns. These issues must be addressed before conducting any analysis.
2. A notable challenge with this dataset is the lack of comprehensive information, leading to numerous assumptions. Efforts will be made to enhance the dataset's coherence.
3. Several categories in the dataset contain unrecorded information:
   - Authors: 25.2% of books lack author names.
   - Categories: 11.7% of books lack any specified category.
   - Pages: 314 books have uncounted pages.
   - Rating: 83.7% of books lack a rating.
   - Publisher: 43.7% of books lack publisher names.
   - Published Year: Approximately 19 books lack information on the year of publication.
4. The dataset dominantly consists of books in English.
5. There may be a correlation between unrecorded data and the years in which books were published. This hypothesis arises from the notion that older books are harder to detect, resulting in incomplete information. Further analysis is required to validate this assumption.
6. Older books may contribute to the absence of complete information about the publisher.
7. An anomaly exists in the maturityRating column, potentially indicating an error in data scraping. This discrepancy requires closer examination to determine its accuracy.
8. This analysis aims to demonstrate the process of cleaning, manipulating, and presenting data through simple visualizations. While it's far from perfect, it highlights the importance of having clean and accurate data for effective visualization, understanding, and presentation.
""")
