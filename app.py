import streamlit as st
import pandas as pd
import re
import random
import plotly.express as px
import zipfile
import os
import logging
import nltk
import datetime
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from nltk.corpus import stopwords
from dateutil.relativedelta import relativedelta
import seaborn as sns
from matplotlib import rcParams

# Put logging on DEBUG MODE
# logging.basicConfig(level=logging.DEBUG)
# Turn off DEBUG MODE
logging.basicConfig(level=logging.CRITICAL)

# --------------------------------------------------------------------------------------------------------------------------------- 

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# set streamlit config as wide
st.set_page_config(
        page_title="Saucy (Home)",
        page_icon= "🔥",
        layout="wide"
    ) 


# Sidebar for navigation
with st.sidebar:
    pages = option_menu(
                        "Menu", 
                        ["Welcome", "More Insights", "coming soon..."], 
                        icons=["house", "bar-chart-line"], 
                        menu_icon="justify",
                        styles={
        "container": {"background-color": "#fafafa"},
        "nav-link": {"font-size": "17px", "text-align": "justify", "margin": "0px", "--hover-color": "#eee"}
                                   }
                            )
st.sidebar.success("Select a Page above")
st.sidebar.warning("Collapse this sidebar for a better viewing experience😊", icon="❕")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload Your Whatsapp Chat (zip files only!)", type="zip")
# Caption below the File Uploader
st.sidebar.caption("Please be rest assured that we don't collect or process your data in anyway.")

# ---------------------------------------------------------------------------------------------------------------------------------

m_start =  0; m_end = 12; a_start = 12; a_end =  17; e_start = 17; e_end = 20

# Define time ranges
@st.cache_data
def categorize_time(time):
    if time >= m_start and time < m_end:
        return 'Morning'
    elif time >= a_start and time < a_end:
        return 'Afternoon'
    elif time >= e_start and time < e_end :
        return 'Evening'
    else:
        return 'Night'
            
# Define week ranges
def get_week_of_month(day):
    week_num = day / 7
    if week_num <= 1:
        return "Week 1"
    elif week_num <= 2:
        return "Week 2"
    elif week_num <= 3:
        return "Week 3"
    else:
        return "Week 4"

# Create function to process data
@st.cache_data
def process_chat_file(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref: 
            # Find a .txt file
            txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
            if txt_files:
                target_file = txt_files[0]  # Extract the first .txt file found
                zip_ref.extract(target_file) 
                logging.debug("Unzipped Successfully")
                text_file_path = target_file
                logging.debug("Done with txt file")
            else:
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(' ')
                with col2:
                    st.image("assets/error.jpeg")
                with col3:
                    st.write(' ')
                st.error("No valid whatsapp chat data found in your uploaded zip file.", icon="🚫")
                st.warning("Please upload a valid whatsapp chat file to proceed with the analysis", icon="❗")
                st.stop()
    except RuntimeError:
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image("assets/error.jpeg")
        with col3:
            st.write(' ')
        st.error("Sorry, we can't process password protected files at the moment", icon="🚫")
        st.warning("Please upload a valid whatsapp chat file to proceed with the analysis", icon="❗")
        st.stop()

    # Open the txt file and read content to a variable
    with open(text_file_path, 'r', encoding='utf-8') as file:
        chat_data = file.readlines()

    # # delete the file
    os.remove(text_file_path)
    logging.debug("Text file deleted")


    # Create and process DataFrame from chat data
    chat_dict = {
    "datetime_str": [],
    "sender": [],
    "message": []
    }

    # regex pattern to use for extraction
    datetime_pattern = r"(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}\s?[APM]{2})"
    sender_pattern = r"- (.*?):"
    message_pattern = r": (.+)"
    logging.info("Regex just ready to go")

    # Process each line
    for line in chat_data:
        date_match = re.search(datetime_pattern, line)
        if date_match:
            chat_dict["datetime_str"].append(date_match.group(1))
            sender_match = re.search(sender_pattern, line)
            message_match = re.search(message_pattern, line)
            
            chat_dict["sender"].append(sender_match.group(1) if sender_match else None)
            chat_dict["message"].append(message_match.group(1) if message_match else line.strip())
        else:
            chat_dict["datetime_str"].append(None)
            chat_dict["sender"].append(None)
            chat_dict["message"].append(line.strip())
        
    logging.debug("Data Dictionary created now, proceeding to DataFrame")
    
    
    # Create DataFrame and process
    df = pd.DataFrame(chat_dict)
    logging.debug("DataFrame created, proceeding to Data Cleaning")

    
      # Forward fill missing values
    df["datetime_str"] = df["datetime_str"].ffill()
    df['datetime_str'] = df['datetime_str'].str.replace("\u202f", "")
    logging.info("Datetime filled down successfully")

    df["sender"] = df["sender"].ffill()
    logging.info("Sender filled down successfully")
    
    
    # Convert the entire datetime column (make sure it's a string column first)
    df["datetime"] = pd.to_datetime(df["datetime_str"], format="%m/%d/%y, %I:%M%p")

    # Extract date components
    df["date"] = df['datetime'].dt.date
    df["datename"] = df['datetime'].dt.strftime("%A, %d %B %Y")
    df['day'] = df['datetime'].dt.day
    df['day_name'] = df['datetime'].dt.strftime("%A")
    df["time"] = df["datetime"].dt.strftime("%I:%M %p")
    df['hour'] = df['datetime'].dt.hour
    df['timecategory'] = df['hour'].apply(categorize_time)
    df['week_of_month'] = df['day'].apply(get_week_of_month)
    df["month"] = df['datetime'].dt.month_name()
    df['year'] = df['datetime'].dt.isocalendar().year
    logging.info("Date Components added")
    
    # Clean the data 
    index_to_drop = df.query("message.str.contains('Tap to learn more.') or message == 'null' or message == '' or message == ' '").index
    df.drop(index=index_to_drop, inplace=True)
    
    logging.info("All done, df is ready!")
    
    return df

# ---------------------------------------------------------------------------------------------------------------------------------


# MAIN OPERATIONS

if uploaded_file == None:
    st.title("Welcome to the Saucy chat Analyzer")
    st.image("assets/saucy.jpeg")
    st.markdown("""
            ### 📱 Analyze Your WhatsApp Chats
            
            Upload your WhatsApp chat export (ZIP file) to:
            - 📊 View detailed chat statistics
            - 👥 Analyze participant behavior
            - 🕒 Track activity patterns
            - 😊 See emoji usage
            - 📈 Visualize trends
            
            #### How to export your chat:
            1. Open WhatsApp chat
            2. Click ⋮ (menu)
            3. More > Export chat
            4. Choose 'Without Media'
            5. Save and upload the ZIP file
        """)
    st.warning("Please upload a chat file to proceed with the analysis")
    st.caption("Use the upload button in the sidebar")
    st.session_state.clear()


else:
    st.sidebar.image("assets/saucy.jpeg")
    
    # Check if df exists in session state
    if 'df' not in st.session_state:
        with st.spinner('Processing your file...\nPlease wait 😁'):
            st.session_state.df = process_chat_file(uploaded_file)
    
    df = st.session_state.df

    if pages == "Welcome":
        st.header("Let's get to insighting!!!🤓🤓")
        ice_breaker = df['sender'].iloc[0]

        # Initialize session state for persistent values
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.options = None

        # Only generate options once when the page first loads
        if not st.session_state.initialized:
            years = list(df["year"].unique())
            years.sort()
            months = list(df['month'].unique())
            month_answer = df["month"].iloc[0]
            months.remove(month_answer)
            options = random.sample(months, k=2)
            options.insert(random.randint(0, 2), month_answer)
            options.insert(0, "Choose...")
            
            # Store values in session state
            st.session_state.options = options
            st.session_state.month_answer = month_answer
            st.session_state.initialized = True

        # Use the stored values for the selectbox
        guess_month = st.selectbox(
            f"Can you guess the month in {df['year'].iloc[0]} the chat started?",
            options=st.session_state.options,
            key='month_selectbox'
        )

        # Check the answer
        if guess_month and guess_month == st.session_state.month_answer:
            st.success(f"YES!\n This entire chat chat started by {df['time'].iloc[0]} on a {df['datename'].iloc[0]} 😋✅")
            difference = relativedelta(datetime.datetime.now(), df["date"].iloc[0])
            st.caption(f"This was {difference.years} years, {difference.months} months, and {difference.days} days ago.")
            st.info(f"{ice_breaker} broke the ice on this chat 💃🏽💃🏽💃🏽🕺🏽🕺🏽🕺🏽")
        elif not guess_month or guess_month == "Choose...":
            st.warning("Pick the correct answer to see the results! 😁")
        elif guess_month and guess_month != st.session_state.month_answer:
            # st.warning("Pick the correct answer to see the results! 😁")
            st.error(f"Not quite! 😁", icon="❌")
       



        times = list(df['hour'].unique())
        times.sort()  # Sort the hours for better presentation
            
        # Calculate message counts and categorize times
        hour_message_count = df['hour'].value_counts().reset_index()
        hour_message_count["timecategory"] = hour_message_count['hour'].apply(categorize_time)

        st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);">
                    """,
                    unsafe_allow_html=True
                    )
        
        st.info("📊 Hourly Message Distribution", icon="ℹ️")
        
        col1, col2 = st.columns([1, 1])
        
        with col2:
            st.write("Here's a breakdown of messages by hour of the day.")
            
            # Get top 3 hours from the hour_message_count we stored in session state
            top_3 = hour_message_count.nlargest(3, "count")
            
            st.markdown("### 🏆 Most Active Hours")
            
            # Create a more engaging display of top hours with their message counts
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                period = "AM" if row['hour'] < 12 else "PM"
                hour_12 = row['hour'] if row['hour'] <= 12 else row['hour'] - 12
                hour_12 = 12 if hour_12 == 0 else hour_12
                
                st.markdown(
                    f"**{i}.** {hour_12}:00 {period} "
                    f"*({row['count']} messages)*"
                )
            
            # Get the most common time category
            category_count = (hour_message_count
                            .groupby('timecategory')['count']
                            .sum()
                            .reset_index()
                            .sort_values('count', ascending=False))
            
            st.markdown("### ⭐ Peak Activity Period")
            st.success(
                f"Most messages are sent during the "
                f"**{category_count['timecategory'].iloc[0]}** ⚡",
                icon="✨"
            )
            
        with col1:
            # Create a more polished bar chart
            fig = px.bar(
                hour_message_count,
                x='hour',
                y='count',
                color='timecategory',
                title="Message Distribution by Hour",
                labels={
                    "count": "Number of Messages",
                    "hour": "Hour of Day (0-23)",
                    'timecategory': "Time of Day"
                },
                template="plotly_dark",
                custom_data=['timecategory']
            )
            
            # Customize the layout
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis={"showgrid": False},
                xaxis={"dtick": 1},  # Show all hour labels
                title={
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            # Add hover template
            fig.update_traces(
                    hovertemplate=(
                        "Hour: %{x}:00<br>" +
                        "Messages: %{y}<br>" +
                        "Time of Day: %{customdata[0]}" +
                        "<extra></extra>"
                )
            )
            
            # Display the chart with custom width
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Feel free to zoom-in, pan and download the chart above")
        st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);">
                    """,
                    unsafe_allow_html=True
                    )


        st.markdown("""
    <style>
    .word-analysis-header {
        text-align: center;
        color: #1f1f1f;
        font-size: 1.5rem;
        background-color: #FFB6C1;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .wordcloud-title {
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

        # Get unique senders
        people = sorted(list(set(df["sender"].dropna())))
        word_counts = {}

        # Process data for each sender
        for sender in people:
            # Filter messages for current sender
            sender_df = df[df["sender"] == sender].copy()
            
            # Remove unwanted messages
            unwanted_messages = [
                'Tap to learn more.',
                'null',
                '',
                ' ',
                '<Media omitted>'
            ]
            sender_df = sender_df[~sender_df['message'].isin(unwanted_messages)]
            
            # Clean and tokenize messages
            all_words = []
            for msg in sender_df['message']:
                # Convert to string, clean, and tokenize
                if isinstance(msg, str):
                    # Remove special characters and numbers
                    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', msg.lower())
                    # Split into words and filter
                    words = [word.strip() for word in cleaned.split() 
                            if word.strip() not in stop_words and len(word.strip()) > 1]
                    all_words.extend(words)
            
            # Calculate word frequencies
            word_counts[sender] = pd.DataFrame(
                Counter(all_words).items(), 
                columns=['words', 'count']
            ).sort_values(by='count', ascending=False)

        # Display header
        st.markdown('<p class="word-analysis-header" style="">Top 50 Most Used Words by Each Participant of the chat</p>', unsafe_allow_html=True)

        # Create columns for word clouds and data
        cols = st.columns(2)

        # Display word clouds and data for each person
        for idx, sender in enumerate(people):
            with cols[idx]:
                # Get top words
                top_words = word_counts[sender].nlargest(50, 'count')
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=50,
                    min_word_length=2  # Ensure no single letters
                ).generate_from_frequencies(
                    dict(zip(top_words['words'], top_words['count']))
                )
                
                # Create and style the plot
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)
                
                # Display word cloud and data
                st.markdown(f'<p class="wordcloud-title">{sender}\'s Most Used Words</p>', unsafe_allow_html=True)
                st.pyplot(fig)
                plt.close()
                
                st.caption("Word frequency breakdown:")
                
                # Style the dataframe
                st.dataframe(
                    top_words,
                    column_config={
                        "words": "Word",
                        "count": "Frequency"
                    },
                    hide_index=True,
                    use_container_width=True
                )

        
        st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);">
                    """,
                    unsafe_allow_html=True
                    )
        
        col1, col2 = st.columns(2)
        with col1:
            # First Plot
            st.markdown("""<h3 style="font-size: 30px; text-align: center; color: green; font-weight: bold;">Total Number of Messages Per User</h3>""", unsafe_allow_html=True)
            senders = df["sender"].value_counts().reset_index()
            sender_fig = px.bar(senders, y="sender", x="count", color="sender", template="plotly_dark", labels={"sender":"Sender", "count":"Total Message"}, text="count")
            sender_fig.update_traces(textposition="inside", insidetextanchor="middle", textfont={"size":17})
            st.plotly_chart(sender_fig)
            st.success(f"{senders['sender'][0]} sent the most message in this chat 🥇")
            
        with col2:
            # Second Plot
            st.markdown("""<h3 style="font-size: 30px; text-align: center; color: green; font-weight: bold;">Top 3 Active Days</h3>""", unsafe_allow_html=True)
            top_3_days = list(df["day_name"].value_counts().nlargest(3).index)
            fav_days = (
                        df.groupby(["day_name", "sender"])[["day_name", "sender"]]
                        .value_counts().reset_index()
                        .sort_values("count")
                        .query("day_name in @top_3_days")
                       )
            fav_days_fig = px.bar(fav_days, y="day_name", x="count", color="sender", template="plotly", labels={"day_name":"Day of the week", "count":"Total Message"}, text="count")
            fav_days_fig.update_traces(textposition="inside", insidetextanchor="middle", textfont={"size":17})
            st.plotly_chart(fav_days_fig)
            st.success(f"{top_3_days[0]}s, {top_3_days[1]}s and {top_3_days[2]}s are the 3 most active days of your chat", icon="✅")
            
            
        st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);">
                    """,
                    unsafe_allow_html=True
                    )

        # third Plot
        st.markdown("""<h3 style="font-size: 30px; text-align: center; color: green; font-weight: bold;">Yearly activity</h3>""", unsafe_allow_html=True)
        yearly_count = list(df["year"].value_counts().nlargest(3).index)
        yearly_activity = (
                    df.groupby(["year", "sender"])[["year", "sender"]]
                    .value_counts().reset_index()
                    .sort_values("count")
                    .query("year in @yearly_count")
                   )
        yearly_activity["year"] = yearly_activity["year"].astype("str")
        yearly_activity_fig = px.bar(yearly_activity, y="year", x="count", color="sender", template="plotly", labels={"year":"Year", "count":"Total Message"}, text="count")
        yearly_activity_fig.update_traces(textposition="inside", insidetextanchor="middle", textfont={"size":17})
        st.plotly_chart(yearly_activity_fig)
        st.success(f"{yearly_count[0]} is the most active year of your chat.", icon="🌟")

        # Navigation hint
        st.info("🚀 Check out the 'More Insights Page' for additional analysis! 📊", icon="ℹ️")
    

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

    elif pages == "More Insights":
        st.header("Here's more Sauce!! 🥳", divider="rainbow")
    
        if df is None or df.empty:
            st.error("Please upload a chat file first!")

        
        people = list(set(df["sender"].dropna()))
    
        if len(people) < 2:
            st.warning("Messages were not exchanged back and forth between two people in this chat.")

            
        # Cache the commonly used data processing
        @st.cache_data
        def process_chat_data(df):
            # Clean the dataframe once
            clean_df = df.copy()
            index_to_drop = clean_df.query("message.str.contains('Tap to learn more.') or message == 'null' or message == '' or message == ' '").index
            clean_df.drop(index=index_to_drop, inplace=True)
            
            # Get basic stats
            ice_breaker = clean_df['sender'].iloc[0]
            
            # Process media stats
            media_stats = pd.DataFrame({
                'Names': people,
                'Total Media': [
                    clean_df[clean_df["sender"] == person]['message'].str.count("<Media omitted>").sum()
                    for person in people
                ]
            })
            
            media_winner = (
                "It's a tie!" 
                if len(set(media_stats['Total Media'])) == 1 
                else media_stats.loc[media_stats['Total Media'].idxmax(), 'Names']
            )
            
            return clean_df, ice_breaker, media_stats, media_winner
        
        clean_df, ice_breaker, media_stats, media_winner = process_chat_data(df)
        
        # Interactive Games Section
        with st.expander("🎮 Fun Guessing Games", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌟 Ice Breaker")
                guess_ice_breaker = st.selectbox(
                    "Who started the conversation?", 
                    ["Choose..."] + people,
                    key="ice_breaker"
                )
                
                if guess_ice_breaker != "Choose...":
                    if guess_ice_breaker == ice_breaker:
                        st.success("🎯 Correct! You're good at this!", icon="✅")
                    else:
                        st.error(f"Not quite! It was {ice_breaker} who broke the ice!", icon="❌")
            
            with col2:
                st.subheader("📸 Media Master")
                guess_media_sender = st.selectbox(
                    "Who sent the most media files?",
                    ["Choose..."] + people,
                    key="media_master"
                )
                
                if guess_media_sender != "Choose...":
                    if guess_media_sender == media_winner:
                        st.success("🎯 Spot on! You know your chat well!", icon="✅")
                    else:
                        st.error(f"Actually, {media_winner} sent the most media!", icon="❌")

        try:
            # Visual Analytics Section
            st.markdown("""---""")
            st.subheader("📊 Chat Analytics Dashboard", divider="violet")
            
            # Time Period Selector
            col1, col2 = st.columns(2)
            with col1:
                year_filter = st.selectbox('📅 Select Year', ['All time'] + sorted(list(set(df['year']))))
            with col2:
                month_filter = st.selectbox("📅 Select Month", ["All"] + sorted(list(set(df['month']))))
            
            # Filter data based on selection
            filtered_df = clean_df.copy()
            if year_filter != 'All time':
                filtered_df = filtered_df[filtered_df['year'] == year_filter]
            if month_filter != 'All':
                filtered_df = filtered_df[filtered_df['month'] == month_filter]
            
            # Key Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_messages = len(filtered_df)
                st.metric("💬 Total Messages", f"{total_messages:,}")
                
            with col2:
                total_media = filtered_df['message'].str.count("<Media omitted>").sum()
                st.metric("📸 Media Shared", f"{total_media:,}")
                
            with col3:
                most_active_hour = filtered_df['hour'].mode().iloc[0]
                st.metric("⏰ Peak Hour", f"{most_active_hour:02d}:00")
                
            with col4:
                deleted_msgs = filtered_df['message'].str.contains('deleted this message').sum()
                st.metric("🗑️ Deleted Messages", f"{deleted_msgs:,}")
                
            with col5:
                edited_msgs = filtered_df['message'].str.contains('<This message was edited>').sum()
                st.metric("✏️ Edited Messages", f"{edited_msgs:,}")
            
            # Advanced Analytics
            st.markdown("""---""")
            tab1, tab2, tab3 = st.tabs(["📈 Time Analysis", "👥 User Activity", "🔤 Content Analysis"])
            
            with tab1:
                # Time-based visualizations
                fig = px.bar(
                    filtered_df.groupby('date')['message'].count().reset_index(),
                    x='date', 
                    y='message',
                    title="Message Frequency Over Time",
                    labels={'message': 'Number of Messages', 'date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Hour activity heatmap
                hour_day_data = filtered_df.groupby(['day_name', 'hour'])['message'].count().reset_index()
                fig = px.density_heatmap(
                    hour_day_data,
                    x='hour',
                    y='day_name',
                    z='message',
                    title="Activity Heatmap by Day & Hour",
                    labels={'message': 'Number of Messages', 'hour': 'Hour of Day', 'day_name': 'Day of Week'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # User activity comparison
                user_stats = pd.DataFrame({
                    'User': people,
                    'Messages': [len(filtered_df[filtered_df['sender'] == person]) for person in people],
                    'Media': [filtered_df[filtered_df['sender'] == person]['message'].str.contains('<Media omitted>').sum() for person in people],
                    'Links': [filtered_df[filtered_df['sender'] == person]['message'].str.contains('http').sum() for person in people]
                })
                
                fig = px.bar(
                    user_stats,
                    x='User',
                    y=['Messages', 'Media', 'Links'],
                    title="User Activity Comparison",
                    barmode='group',
                    text="value"
                )
                fig.update_traces(textposition="inside", textfont={"size":13})
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                # Word cloud and common phrases
                if not filtered_df.empty:
                    from collections import Counter
                    import re
                    
                    # Word frequency analysis
                    def clean_text(text):
                        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                        return text.split()
                    
                    words = [
                        word 
                        for msg in filtered_df['message'] 
                        for word in clean_text(str(msg))
                        if len(word) > 3 and word not in stop_words
                    ]
                    
                    word_freq = Counter(words).most_common(10)
                    word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                    
                    fig = px.bar(
                        word_df,
                        x='Word',
                        y='Frequency',
                        title="Most Common Words",
                        color='Frequency',
                        text="Frequency"
                    )
                    fig.update_traces(textposition="inside", insidetextanchor="middle")
                    st.plotly_chart(fig, use_container_width=True)
            
        except:
            st.info("Please select a range of month/date present in the chat at the moment 😊😊.")


    else:
        st.header("Please check back later for more updates! 😁😁😁")






















        
        
