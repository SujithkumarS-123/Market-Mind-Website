MARKET MIND — LOCAL WEBSITE (STREAMLIT)
=======================================

This is a separate local website wrapper for the saved Market Mind V2 model. It does NOT retrain, overwrite, or edit the original Colab notebook/model.

IMPORTANT: the ZIP does not include your model files. Copy these four existing files from Google Drive into the `artifacts` folder before launching:

1. Market_Mind_25_Stock_CNN_LSTM_V2.keras
2. Market_Mind_Preprocessing.joblib
3. Market_Mind_V2_Target_Scaler.joblib
4. Market_Mind_Stock_Mapping.joblib

Expected Google Drive source folder:
MyDrive/Market_Mind_Project/outputs/

Copy those files into:
market_mind_local_website/artifacts/

If Market_Mind_Stock_Mapping.joblib is in a different Drive folder, copy it from its actual saved location. Do not substitute a different scaler or retrain the model.

HOW TO RUN ON WINDOWS
---------------------
1. Install Python 3.11 (recommended for compatibility with TensorFlow).
2. Extract the ZIP to a normal folder, for example Desktop\market_mind_local_website.
3. Put the four artifacts in the `artifacts` folder.
4. Open Command Prompt in the extracted folder.
5. Create and activate a virtual environment:

   py -3.11 -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py

6. Streamlit will show a local URL, normally:

   http://localhost:8501

Open that address in your browser. Keep the terminal open while using the local website.

FEATURES
--------
- Dark stock-market-style interface
- Dropdown for the saved model's stock mapping (25 NSE stocks expected)
- Fetches completed daily market data using yfinance when a prediction is requested
- Reuses saved CNN-LSTM and preprocessing artifacts
- Displays latest close, predicted next-session close, estimated change and direction
- Interactive candlestick, volume, moving-average and historical actual-vs-predicted charts
- Historical evaluation metrics and downloadable evaluation CSV

NOTES / LIMITATIONS
-------------------
- Internet access is required for market data.
- The backend runs locally on your computer; Colab does not need to stay running.
- This is a local website, not yet a public deployed website.
- First installation can take time because TensorFlow is large.
- The feature formulas are copied from the final dashboard cell in the notebook. The original training pipeline should be verified against these formulas before relying on model results.
- Forecasts are experimental, may be inaccurate, and are not investment advice.
