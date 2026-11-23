import nltk

for resource in ("wordnet", "omw-1.4"):
    nltk.download(resource)

print("NLTK resources are ready.")
