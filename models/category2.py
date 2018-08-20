from mongoengine import Document,StringField,IntField

class Category2(Document):
    primary=IntField(primary_key=True)
    name=StringField()