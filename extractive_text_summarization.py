# import some modules
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

stopwords = list(STOP_WORDS)
stopwords

text = """In functional programming, a monad is a software design pattern with a structure that combines program 
fragments (functions) and wraps their return values in a type with additional computation. In addition to defining a 
wrapping monadic type, monads define two operators: one to wrap a value in the monad type, and another to compose 
together functions that output values of the monad type (these are known as monadic functions). General-purpose 
languages use monads to reduce boilerplate code needed The term "monad" in programming actually goes all the way back 
to the APL and J programming languages, which do tend toward being purely functional. However, in those languages, 
"monad" is only shorthand for a function taking one parameter (a function with two parameters being a "dyad", 
and so on).[19] The mathematician Roger Godement was the first to formulate the concept of a monad (dubbing it a 
"standard construction") in the late 1950s, though the term "monad" that came to dominate was popularized by 
category-theorist Saunders Mac Lane.[citation needed] The form defined above using bind, however, was originally 
described in 1965 by mathematician Heinrich Kleisli in order to prove that any monad could be characterized as an 
adjunction between two (covariant) functors.[20] 

Starting in the 1980s, a vague notion of the monad pattern began to surface in the computer science community. 
According to programming language researcher Philip Wadler, computer scientist John C. Reynolds anticipated several 
facets of it in the 1970s and early 1980s, when he discussed the value of continuation-passing style, category theory 
as a rich source for formal semantics, and the type distinction between values and computations.[4] The research 
language Opal, which was actively designed up until 1990, also effectively based I/O on a monadic type, 
but the connection was not realized at the time.[21] 

The computer scientist Eugenio Moggi was the first to explicitly link the monad of category theory to functional 
programming, in a conference paper in 1989,[22] followed by a more refined journal submission in 1991. In earlier 
work, several computer scientists had advanced using category theory to provide semantics for the lambda calculus. 
Moggi's key insight was that a real-world program is not just a function from values to other values, but rather a 
transformation that forms computations on those values. When formalized in category-theoretic terms, this leads to 
the conclusion that monads are the structure to represent these computations.[3] 

Several others popularized and built on this idea, including Philip Wadler and Simon Peyton Jones, both of whom were 
involved in the specification of Haskell. In particular, Haskell used a problematic "lazy stream" model up through 
v1.2 to reconcile I/O with lazy evaluation, until switching over to a more flexible monadic interface.[23] The 
Haskell community would go on to apply monads to many problems in functional programming, and in the 2010s, 
researchers working with Haskell eventually recognized that monads are applicative functors;[24][i] and that both 
monads and arrows are monoids.[26] 

At first, programming with monads was largely confined to Haskell and its derivatives, but as functional programming 
has influenced other paradigms, many languages have incorporated a monad pattern (in spirit if not in name). 
Formulations now exist in Scheme, Perl, Python, Racket, Clojure, Scala, F#, and have also been considered for a new 
ML standard. """
# make sure the text is long enough!

# load nlp model
nlp = spacy.load("en_core_web_sm")

# Turn the original text into a spacy doc object
doc = nlp(text)

# create tokens for each word
tokens = [token.text for token in doc]

# clean out stopwords and newline characters (/n) from the tokens

cleaned = [word for word in tokens if word not in stopwords and word not in punctuation + '/n']
# for every single word in the tokens list, we are going to add it to this list


# let's first do extractive summarization
# the core of extractive summarization is 1) create a dictionary for word frequencies
# and 2) create word scores

word_frequencies = {}

for word in cleaned:
  if word not in word_frequencies.keys():
    word_frequencies[word] = 1
  else:
    word_frequencies[word] += 1


# normalize frequency (make all values between 0-1)

# we find the max_frequency to use as a benchmark
max_frequency = max(word_frequencies.values())

# now we do the actual normalizing
for key in word_frequencies:
  word_frequencies[key] /= max_frequency


# create sentence tokens

sentence_tokens = [sent for sent in doc.sents]

# give sentence each sentence a score based on the words inside that sentence

sentence_scores = {}

for sent in sentence_tokens:
  for word in sent:
    if word.text.lower() in word_frequencies:
      # now we know the word has a score in the frequencies dictionary
      if sent not in sentence_scores.keys():
        sentence_scores[sent] = word_frequencies[word.text.lower()]
      else:
        sentence_scores[sent] += word_frequencies[word.text.lower()]


# select the top sentences with highest scores

from heapq import nlargest

select_length = 4
summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

# create final summary

final_summary = [word.text for word in summary]
print("\n".join(final_summary))