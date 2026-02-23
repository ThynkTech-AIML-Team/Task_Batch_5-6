def extract_topics(model, feature_names, n_top_words=10):

    topics = []

    for topic in model.components_:

        words = [
            feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]
        ]

        topics.append(words)

    return topics
