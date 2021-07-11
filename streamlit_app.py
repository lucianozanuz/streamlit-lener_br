import pandas as pd
import numpy as np
import streamlit as st

# Add a title
st.title('Título da aplicação')
# Add some text
st.text('Texto')

st.header('Header da aplicação.')
st.subheader('Subheader da aplicação')
st.text('Texto: Upload excel files with only one column, even if you put multiple columns only the first one will be used')


with st.form("my_form"):
   st.write("Inside the form")
   slider_val = st.slider("Form slider")
   checkbox_val = st.checkbox("Form checkbox")

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("slider", slider_val, "checkbox", checkbox_val)

st.write("Outside the form")



file_lookup = st.file_uploader("Lookup list", help="List with values to be matched, in the Left-join that's the left side")
file_match = st.file_uploader("Match list", help="List with values to match with, in the Left-join that's the right side")




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

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)    
    
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
