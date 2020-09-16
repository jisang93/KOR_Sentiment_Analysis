from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

model_output_path = '/home/lab12/workspace/Korean_Sentiment/KOR_Sentiment_Analysis/Project_Estimation/model'

model = AutoModelForSequenceClassification.from_pretrained(model_output_path)
tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-large')

sentiment_classifier = pipeline('labella_model_sentiment', model=model, tokenizer=tokenizer, framework='pt')

reviews = ["홍선영이땜에 광주까지 싫어질려고 함 진짜 매우새야 제발 홍선영이 좀 나오게 하지마라",
           "자매들이  되게  재밌고  귀엽네!!!" ]

results = sentiment_classifier(reviews)

for result in results:
    print(f"label : {result['label']}, with score : {round(result['score'], 4)}")