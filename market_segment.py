import streamlit as st
import pandas as pd
import pickle

# Set the title of the app
st.title("Market Segmentation App")

from PIL import Image
st.subheader("Hello!    Welcome to Product Recommendation")
# st.set_page_config(page_title = 'Welcome to Product Recommendation', layout='centered')
# st.title("Customer Segment App")
image = Image.open("MarketSegment.png")
st.image(image,use_column_width = True)


#load the model
classifier = pickle.load(open('final_pipeline.pkl','rb'))


# customer segmentation function
def segment_customers(input_data):
    
    prediction=classifier.predict(pd.DataFrame(input_data, columns=['State', 'Market', 'Market Size', 'Product Type', 'Type']))
    output_str = prediction[0]
    return output_str

def main():
    
   
    State = st.text_input("Type In The State",type='default')
    Market = st.radio ( "Select Market", ('East', 'West','South','Central') )
    Market_Size = st.radio ( "Select Market Size", ('Small Market', 'Major Market') )
    Product_Type = st.radio ( "Select Product Type", ("Coffee","Tea","Espresso","Herbal Tea") )
    Type = st.radio ( "Please Select The Type", ('Regular', 'Decaf') )
    
    
    result = ""

    # when button is clicked, make the prediction and store it
    if st.button("Recommend Product"):
        result=segment_customers([[State,Market,Market_Size,Product_Type,Type]])
    
    st.success(result)
    

if __name__ == '__main__':
        main ()


