!pip install spacy==3
!python -m spacy download en_core_web_trf

import spacy

# Load the language model
nlp = spacy.load('en_core_web_trf')

# Sample text for demonstration
text = """
        The rise of technology has drastically changed the way we live our lives. From smartphones to social media, technology has revolutionized the way we communicate and interact with each other. While technology has brought many benefits, it has also raised concerns about privacy and the impact on our mental health. As we continue to advance in technology, it is important to consider both the positive and negative effects on society." 
        """

# Process the text with spaCy
doc = nlp(text)

# Calculate sentence scores based on their similarity to the document
sent_scores = []
for sent in doc.sents:
    score = 0
    for word in sent:
        score += word.similarity(doc)
    sent_scores.append((sent, score))

# Sort the sentences in decreasing order of score and get the top 3
sent_scores = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:3]

# Extract the top 3 sentences and join them to form a summary
summary = ' '.join([sent[0].text for sent in sent_scores])

# Print the summary
print(summary)
