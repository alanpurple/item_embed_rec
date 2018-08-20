from mongoengine import Document,StringField,IntField,FloatField,ListField,ReferenceField
from models import Category1
from models import Category2

class DealW2v(Document):
    meta={'collection':'dealw2v'}
    primary=IntField(primary_key=True)
    title=StringField()
    words=ListField(StringField())
    vectorizedWords=ListField(FloatField())
    category1=ReferenceField(Category1)
    category2=ReferenceField(Category2)