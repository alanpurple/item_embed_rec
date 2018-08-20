from mongoengine import Document,StringField,IntField

class Category1(Document):
    primary=IntField(primary_key=True)
    name=StringField()