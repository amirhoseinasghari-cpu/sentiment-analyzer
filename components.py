"""
UI Components for Persian Sentiment Analyzer
"""

import streamlit as st


def render_header():
    """Render application header"""
    st.markdown("""
        <div style="text-align: center; padding: 30px 0;">
            <h1 style="font-size: 3em; margin-bottom: 10px;">
                🎭 تحلیلگر احساسات فارسی
            </h1>
            <p style="font-size: 1.2em; color: #666;">
                Persian Sentiment Analysis powered by AI
            </p>
            <div style="margin-top: 20px;">
                <span style="background: #667eea; color: white; padding: 5px 15px; 
                             border-radius: 20px; margin: 0 5px;">🤖 BERT</span>
                <span style="background: #00b894; color: white; padding: 5px 15px; 
                             border-radius: 20px; margin: 0 5px;">⚡ Real-time</span>
                <span style="background: #e84393; color: white; padding: 5px 15px; 
                             border-radius: 20px; margin: 0 5px;">🇮🇷 Persian</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_metrics():
    """Render key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 دقت مدل", "94.5%", "+2.3%")
    with col2:
        st.metric("⚡ سرعت", "<100ms", "-15ms")
    with col3:
        st.metric("📊 پشتیبانی", "3 کلاس", "")
    with col4:
        st.metric("🔤 زبان", "فارسی", "")


def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>ساخته شده با ❤️ | Powered by 
               <a href="https://huggingface.co/HooshvareLab" target="_blank">HooshvareLab</a> & 
               <a href="https://streamlit.io" target="_blank">Streamlit</a>
            </p>
            <p style="font-size: 0.9em;">
                🌟 <a href="https://github.com/yourusername/persian-sentiment-analyzer" target="_blank">
                    مشاهده در GitHub
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_example_cards():
    """Render example text cards"""
    st.markdown("### 📝 نمونه متن‌ها")
    
    examples = [
        {
            'text': 'این رستوران غذای فوق‌العاده‌ای دارد، حتماً پیشنهاد می‌کنم!',
            'sentiment': 'positive',
            'icon': '😊'
        },
        {
            'text': 'کیفیت محصول افتضاح بود، پولم رو دور ریختم.',
            'sentiment': 'negative',
            'icon': '😠'
        },
        {
            'text': 'محصول معمولی بود، نه خوب نه بد.',
            'sentiment': 'neutral',
            'icon': '😐'
        }
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i]:
            st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; 
                            border-right: 4px solid {'#00b894' if example['sentiment'] == 'positive' else '#e74c3c' if example['sentiment'] == 'negative' else '#636e72'};">
                    <p style="direction: rtl; text-align: right;">{example['text']}</p>
                    <p style="text-align: left; font-size: 1.5em;">{example['icon']}</p>
                </div>
            """, unsafe_allow_html=True)