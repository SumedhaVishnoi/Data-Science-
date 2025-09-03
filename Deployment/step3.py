'''
Step 3: Deploy Flask App to Heroku
1. Install Heroku CLI

Download & install from 👉 Heroku CLI

Login:

heroku login

2. Project Structure

Your folder should look like this:

project/
│── app.py
│── iris_model.pkl
│── requirements.txt
│── Procfile

3. Create requirements.txt

Run:

pip freeze > requirements.txt


👉 Make sure it contains at least:

flask
numpy
scikit-learn

4. Create Procfile (no extension, just Procfile)

Content:

web: python app.py

5. Initialize Git & Commit Code
git init
git add .
git commit -m "Initial commit"

6. Create Heroku App & Deploy
heroku create iris-flask-api-demo
git push heroku master

7. Open App
heroku open


👉 Your API is now live 🌍 at
https://iris-flask-api-demo.herokuapp.com/'''