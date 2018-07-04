from mongoengine import Document,StringField,IntField,FloatField,ListField

class DealW2v(Document):
    meta={'collection':'dealw2v'}
    primary=IntField(primary_key=True)
    words=ListField(StringField())
    vectorizedWords=ListField(FloatField())