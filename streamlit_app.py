import pandas as pd
import numpy as np
import streamlit as st

# Add a title
st.title('My first app')
# Add some text
st.text('Streamlit is great')


if st.checkbox('Show dataframe'):  
    
    st.dataframe(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    }))

    st.write(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    }))

    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

    
        
    df = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])

    column = st.selectbox(
        'What column to you want to display',
         df.columns)

    st.line_chart(df[column])
    
    df2 = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])

    columns = st.multiselect(
        label='What column to you want to display', options=df.columns)

    st.line_chart(df2[columns])
    
    
    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

    st.map(map_data)

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)

@st.cache
def fetch_and_clean_data():
    df = pd.read_csv('<some csv>')
    # do some cleaning
    return df

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

left_column, right_column = st.beta_columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
