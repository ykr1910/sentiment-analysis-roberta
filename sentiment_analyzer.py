from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

# VADER setup
sia = SentimentIntensityAnalyzer()

# RoBERTa setup
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
sent_pipeline = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL, device=-1)
LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

def analyze_vader(text):
    return sia.polarity_scores(text)

def analyze_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': float(scores[0]),
        'roberta_neu': float(scores[1]),
        'roberta_pos': float(scores[2])
    }

def analyze_roberta_label(text):
    result = sent_pipeline(text)[0]
    mapped_label = LABELS.get(int(result['label'].split("_")[-1]), result['label'])
    return {'label': mapped_label, 'score': float(result['score'])}

def analyze_sentiment(text):
    vader = analyze_vader(text)
    roberta_scores = analyze_roberta(text)
    roberta_label = analyze_roberta_label(text)
    return {
        "vader_neg": vader["neg"],
        "vader_neu": vader["neu"],
        "vader_pos": vader["pos"],
        "vader_compound": vader["compound"],
        **roberta_scores,
        "roberta_label": roberta_label["label"],
        "roberta_label_score": roberta_label["score"]
    }
