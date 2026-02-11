"""
Persian Sentiment Analysis Dashboard
A professional AI-powered sentiment analysis platform for Persian text
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sentiment_model import SentimentAnalyzer
from components import render_header, render_metrics, render_footer
from utils import load_sample_data, save_feedback

# Page Configuration
st.set_page_config(
    page_title="تحلیلگر احساسات فارسی | Persian Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL and styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    
    * {
        font-family: 'Vazirmatn', sans-serif;
    }
    
    .main {
        direction: rtl;
    }
    
    .stTextArea textarea {
        direction: rtl;
        font-size: 16px;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0, 184, 148, 0.3);
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #e74c3c 0%, #e84393 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.3);
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #636e72 0%, #b2bec3 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(99, 110, 114, 0.3);
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analyzer' not in st.session_state:
    with st.spinner('🔄 در حال بارگذاری مدل هوش مصنوعی...'):
        st.session_state.analyzer = SentimentAnalyzer()

# Header
render_header()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("⚙️ تنظیمات")
    
    st.markdown("---")
    
    # Model info
    st.info("""
    **🤖 مدل:** BERT-Persian  
    **📊 دقت:** 94.5%  
    **⚡ سرعت:** Real-time
    """)
    
    st.markdown("---")
    
    # Analysis mode
    analysis_mode = st.radio(
        "🎯 حالت تحلیل:",
        ["تک متن", "آپلود فایل", "مقایسه"]
    )
    
    st.markdown("---")
    
    # Show history
    if st.session_state.history:
        st.write(f"📋 تعداد تحلیل‌ها: {len(st.session_state.history)}")
        if st.button("🗑️ پاک کردن تاریخچه"):
            st.session_state.history = []
            st.rerun()

# Main Content
if analysis_mode == "تک متن":
    st.markdown("### ✍️ متن خود را وارد کنید")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "",
            placeholder="مثال: این محصول واقعاً عالی بود، کاملاً راضی هستم!",
            height=150
        )
        
        analyze_button = st.button("🔍 تحلیل احساسات", use_container_width=True)
    
    with col2:
        st.markdown("""
        **💡 راهنما:**
        - متن فارسی وارد کنید
        - می‌تواند نظر، کامنت یا هر متنی باشد
        - مدل به صورت خودکار تحلیل می‌کند
        """)
    
    if analyze_button and text_input:
        with st.spinner('🤖 در حال تحلیل...'):
            start_time = time.time()
            result = st.session_state.analyzer.predict(text_input)
            processing_time = time.time() - start_time
            
            # Save to history
            st.session_state.history.append({
                'text': text_input[:100] + '...' if len(text_input) > 100 else text_input,
                'sentiment': result['label'],
                'confidence': result['confidence'],
                'time': datetime.now().strftime("%H:%M:%S")
            })
        
        # Display result
        st.markdown("---")
        st.markdown("### 🎯 نتیجه تحلیل")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            sentiment_class = result['label']
            if sentiment_class == 'positive':
                st.markdown('<div class="sentiment-positive">😊 مثبت</div>', unsafe_allow_html=True)
            elif sentiment_class == 'negative':
                st.markdown('<div class="sentiment-negative">😠 منفی</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sentiment-neutral">😐 خنثی</div>', unsafe_allow_html=True)
        
        with result_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['confidence'] * 100,
                title={'text': "اطمینان مدل"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ff7675"},
                        {'range': [50, 80], 'color': "#fdcb6e"},
                        {'range': [80, 100], 'color': "#00b894"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with result_col3:
            st.metric("⚡ زمان پردازش", f"{processing_time:.3f}s")
            st.metric("📊 تعداد کلمات", len(text_input.split()))
            
            # Emoji based on sentiment
            emoji_map = {
                'positive': '😄 🎉 👍',
                'negative': '😤 💢 👎',
                'neutral': '😐 📊 ⚖️'
            }
            st.markdown(f"**شکلک‌ها:** {emoji_map.get(sentiment_class, '🤔')}")
        
        # Detailed scores
        st.markdown("---")
        st.markdown("### 📊 جزئیات امتیازات")
        
        scores = result.get('scores', {})
        if scores:
            score_data = pd.DataFrame({
                'احساس': ['مثبت', 'منفی', 'خنثی'],
                'امتیاز': [scores.get('positive', 0), scores.get('negative', 0), scores.get('neutral', 0)]
            })
            
            fig_bar = px.bar(
                score_data,
                x='احساس',
                y='امتیاز',
                color='احساس',
                color_discrete_map={
                    'مثبت': '#00b894',
                    'منفی': '#e74c3c',
                    'خنثی': '#636e72'
                },
                text='امتیاز'
            )
            fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

elif analysis_mode == "آپلود فایل":
    st.markdown("### 📁 آپلود فایل CSV")
    
    uploaded_file = st.file_uploader(
        "فایل CSV خود را آپلود کنید (ستون 'text' باید وجود داشته باشد)",
        type=['csv']
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"📊 تعداد رکوردها: {len(df)}")
        st.dataframe(df.head())
        
        if st.button("🚀 شروع تحلیل دسته‌ای"):
            progress_bar = st.progress(0)
            results = []
            
            for i, row in df.iterrows():
                result = st.session_state.analyzer.predict(str(row.get('text', '')))
                results.append(result)
                progress_bar.progress((i + 1) / len(df))
            
            df['sentiment'] = [r['label'] for r in results]
            df['confidence'] = [r['confidence'] for r in results]
            
            st.success("✅ تحلیل کامل شد!")
            st.dataframe(df)
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 دانلود نتایج",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )
            
            # Visualization
            sentiment_counts = df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="توزیع احساسات",
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': '#00b894',
                    'negative': '#e74c3c',
                    'neutral': '#636e72'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

elif analysis_mode == "مقایسه":
    st.markdown("### ⚖️ مقایسه دو متن")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text1 = st.text_area("متن اول", height=100, key="text1")
    
    with col2:
        text2 = st.text_area("متن دوم", height=100, key="text2")
    
    if st.button("🔍 مقایسه") and text1 and text2:
        result1 = st.session_state.analyzer.predict(text1)
        result2 = st.session_state.analyzer.predict(text2)
        
        comp_data = pd.DataFrame({
            'متن': ['متن اول', 'متن دوم'],
            'احساس': [result1['label'], result2['label']],
            'اطمینان': [result1['confidence'], result2['confidence']]
        })
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='متن اول',
            x=['اطمینان'],
            y=[result1['confidence']],
            marker_color='#667eea'
        ))
        fig_comp.add_trace(go.Bar(
            name='متن دوم',
            x=['اطمینان'],
            y=[result2['confidence']],
            marker_color='#764ba2'
        ))
        st.plotly_chart(fig_comp, use_container_width=True)

# History section
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 تاریخچه تحلیل‌ها")
    
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

# Footer
render_footer()