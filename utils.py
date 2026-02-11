"""
Utility functions for Persian Sentiment Analyzer
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def load_sample_data() -> pd.DataFrame:
    """
    Load sample Persian text data for testing
    
    Returns:
        DataFrame with sample reviews
    """
    sample_data = {
        'text': [
            'این محصول واقعاً عالی بود، کاملاً راضی هستم!',
            'کیفیت خیلی پایین بود، اصلاً پیشنهاد نمی‌کنم.',
            'قیمت مناسب است ولی کیفیت متوسط.',
            'عالی! بهترین خریدم بود.',
            'اصلاً خوب نبود، پشیمون شدم.',
            'معمولی بود، انتظار بیشتری داشتم.',
            'فوق‌العاده! حتماً دوباره می‌خرم.',
            'بدرد نمی‌خوره، پولتونو دور نریزید.',
            'خوب بود ولی ارسال دیر انجام شد.',
            'عاشق این محصول شدم! ❤️'
        ],
        'category': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'neutral', 'positive', 'negative', 'neutral', 'positive'
        ]
    }
    return pd.DataFrame(sample_data)


def save_feedback(text: str, predicted_sentiment: str, user_feedback: str, 
                  confidence: float, filename: str = "feedback.json") -> None:
    """
    Save user feedback for model improvement
    
    Args:
        text: Input text
        predicted_sentiment: Model prediction
        user_feedback: User's feedback (correct/incorrect)
        confidence: Model confidence score
        filename: Feedback file name
    """
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text,
        'predicted_sentiment': predicted_sentiment,
        'user_feedback': user_feedback,
        'confidence': confidence
    }
    
    feedback_path = Path(filename)
    
    if feedback_path.exists():
        with open(feedback_path, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []
    
    feedback_data.append(feedback_entry)
    
    with open(feedback_path, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)


def analyze_text_statistics(text: str) -> Dict:
    """
    Analyze text statistics
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    words = text.split()
    chars = len(text)
    
    # Persian character detection
    persian_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    
    return {
        'word_count': len(words),
        'char_count': chars,
        'persian_char_count': persian_chars,
        'avg_word_length': chars / len(words) if words else 0,
        'persian_ratio': persian_chars / chars if chars > 0 else 0
    }


def format_confidence(confidence: float) -> str:
    """
    Format confidence score with color indicator
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted string
    """
    percentage = confidence * 100
    if percentage >= 80:
        return f"🟢 {percentage:.1f}%"
    elif percentage >= 50:
        return f"🟡 {percentage:.1f}%"
    else:
        return f"🔴 {percentage:.1f}%"


def get_sentiment_emoji(sentiment: str) -> str:
    """
    Get emoji for sentiment
    
    Args:
        sentiment: Sentiment label
        
    Returns:
        Emoji string
    """
    emoji_map = {
        'positive': '😊',
        'negative': '😠',
        'neutral': '😐'
    }
    return emoji_map.get(sentiment, '🤔')


def export_results(results: List[Dict], format: str = 'csv') -> str:
    """
    Export analysis results
    
    Args:
        results: List of analysis results
        format: Export format (csv/json)
        
    Returns:
        File content as string
    """
    df = pd.DataFrame(results)
    
    if format == 'csv':
        return df.to_csv(index=False, encoding='utf-8')
    elif format == 'json':
        return df.to_json(orient='records', force_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")