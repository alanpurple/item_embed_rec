from mongoengine import Document,StringField,IntField

class Category1(Document):
    meta={'collection':'category1'}
    primary=IntField(primary_key=True)
    name=StringField()