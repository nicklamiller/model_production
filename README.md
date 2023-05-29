# Model Production
A repo to practice simple productionizing of ML models.

This repo is based on the work done in [this TDS article](https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2) by GreekDataGuy.

To summarize the work outlined in the article:

1. A machine learning model is created
2. That machine learning model is pickled for later use
3. A Flask app is created that can then be deployed, incorporating the pickled model and new user input.

In my case, I just ran the flask app locally. In the article the app is additionally deployed on Heroku, but I believe the article was written when Heroku still had free options 😅. Either way, deploying to Heroku is only a couple commands away if one has the resources on their Heroku account to do so.
