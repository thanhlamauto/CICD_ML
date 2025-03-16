import gradio as gr
import pandas as pd
import numpy as np
import skops.io as sio
import datetime
from datetime import timedelta
import os
from skops.io import get_untrusted_types
import joblib


# Load the trained pipeline
def load_model():
    try:
        # Change file path to .joblib extension
        model_path = "/Users/thanhlamnguyen/CICD/model/store_sales_pipeline.joblib"
        if not os.path.exists(model_path):
            return f"Error: Model file not found at {model_path}"
        
        # Use joblib instead of skops
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        return f"Error loading model: {str(e)}"

# Create a function to make predictions
def predict_sales(
    date, 
    store_nbr, 
    family, 
    onpromotion, 
    oil_price, 
    city, 
    state, 
    store_type, 
    cluster
):
    try:
        # Convert inputs to appropriate format
        input_date = pd.to_datetime(date)
        
        # Create features
        features = {
            'date': input_date,
            'store_nbr': int(store_nbr),
            'family': family,
            'onpromotion': int(onpromotion),
            'oil_price': float(oil_price),
            'city': city,
            'state': state,
            'type': store_type,
            'cluster': int(cluster),
            'dcoilwtico': float(oil_price),  # Duplicate field (based on common feature names)
            'year': input_date.year,
            'month': input_date.month,
            'day': input_date.day,
            'dayofweek': input_date.dayofweek,
            'weekend': 1 if input_date.dayofweek >= 5 else 0,
        }
        
        # Load model
        pipeline = load_model()
        if isinstance(pipeline, str):
            return pipeline  # Return error message
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = pipeline.predict(input_df)
        
        # Convert prediction back to original scale (if using log transform)
        final_prediction = np.expm1(prediction)[0]
        
        return f"Predicted Sales: {final_prediction:.2f}"
    
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Create lists of options for dropdowns
store_families = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 
                 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 
                 'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN', 'LADIESWEAR', 
                 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 
                 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 
                 'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD']

store_cities = ['Quito', 'Guayaquil', 'Cuenca', 'Ambato', 'Machala', 'Loja', 'Santo Domingo', 'Manta']
store_states = ['Pichincha', 'Guayas', 'Azuay', 'Tungurahua', 'El Oro', 'Loja', 'Santo Domingo', 'Manabi']
store_types = ['A', 'B', 'C', 'D', 'E']

# Create Gradio interface
with gr.Blocks(title="Store Sales Prediction") as demo:
    gr.Markdown("# Store Sales Prediction Model")
    gr.Markdown("Enter the required information to predict store sales")
    
    with gr.Row():
        with gr.Column():
            date_input = gr.DateTime(label="Date")
            store_nbr = gr.Slider(label="Store Number", minimum=1, maximum=54, step=1, value=1)
            family = gr.Dropdown(label="Product Family", choices=store_families, value=store_families[0])
            onpromotion = gr.Checkbox(label="On Promotion")
            oil_price = gr.Number(label="Oil Price", value=50.0)
        
        with gr.Column():
            city = gr.Dropdown(label="City", choices=store_cities, value=store_cities[0])
            state = gr.Dropdown(label="State", choices=store_states, value=store_states[0])
            store_type = gr.Dropdown(label="Store Type", choices=store_types, value=store_types[0])
            cluster = gr.Slider(label="Cluster", minimum=1, maximum=17, step=1, value=1)
    
    predict_button = gr.Button("Predict Sales")
    output = gr.Textbox(label="Prediction Result")
    
    predict_button.click(
        fn=predict_sales,
        inputs=[date_input, store_nbr, family, onpromotion, oil_price, city, state, store_type, cluster],
        outputs=output
    )
    
    gr.Markdown("### How to use")
    gr.Markdown("""
    1. Enter the date for the prediction
    2. Select the store number
    3. Choose the product family
    4. Indicate if the product is on promotion
    5. Enter the current oil price
    6. Select the store's city and state
    7. Choose the store type and cluster
    8. Click 'Predict Sales' to see the result
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()