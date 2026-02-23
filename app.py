from flask import Flask, request, render_template_string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = Flask(__name__)

html = """
<h2>Resume Screening AI</h2>

<form method="post">
Job Description:<br>
<textarea name="jd" rows="6" cols="60"></textarea><br><br>

Resume Text:<br>
<textarea name="resume" rows="6" cols="60"></textarea><br><br>

<input type="submit">
</form>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        jd = request.form["jd"]
        resume = request.form["resume"]

        documents = [jd, resume]

        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(documents)

        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        match_percentage = round(float(score[0][0]) * 100, 2)

        if match_percentage > 50:
            result = "✅ Strong Match"
        else:
            result = "❌ Low Match"

        return f"<h3>Match Score: {match_percentage}%</h3><h3>{result}</h3>"

    return render_template_string(html)

if __name__ == "__main__":
    app.run(debug=True)
