# 📈 Stock Predictor — Free Hostable App

A lightweight stock prediction web app built with **Streamlit**, **scikit-learn**, and **Alpha Vantage API**. This app allows users to select stocks, view real-time data, and get short-term price forecasts.

---

## 🚀 Features

* **Real-time Stock Data** from Alpha Vantage API
* **Machine Learning Predictions** using Random Forest Regressor
* **Interactive Charts** (Plotly)
* **CSV Data Download**
* Works for **Indian NSE** & **US Stocks**
* **Free Hosting** via Streamlit Cloud or Hugging Face Spaces

---

## 🛠 Installation (Local)

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/stock-predictor-app.git
cd stock-predictor-app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your **Alpha Vantage API key** in a `.env` or Streamlit secrets.
4. Run locally:

```bash
streamlit run app.py
```

---

## 🌐 Free Hosting (Streamlit Cloud)

1. Push your project to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) → **New app**.
3. Select your repo and branch.
4. Set your **Secrets** in **App Settings**:

```toml
ALPHA_VANTAGE_API_KEY = "your_api_key_here"
```

5. Click **Deploy**.

Your app will be live at:

```
https://YOUR_USERNAME-stock-predictor.streamlit.app
```

---

## ⚠️ Disclaimer

This app is for **educational purposes only** and does not constitute financial advice. Use at your own risk.

---

**Built with ❤️ using Streamlit & Alpha Vantage API**
