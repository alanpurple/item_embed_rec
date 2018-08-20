from mongoengine import Document,StringField,IntField

class Category2(Document):
    meta={'collection':'category2'}
    primary=IntField(primary_key=True)
    name=StringField()