from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class PredictionForm(FlaskForm):

    picture=FileField('Submit picture',validators=[FileAllowed(['jpg'])])

    submit = SubmitField('Predict Breed')