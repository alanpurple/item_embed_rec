from mongoengine import Document,StringField,IntField,FloatField,ObjectIdField,ListField,DictField,ReferenceField,EmbeddedDocument

class WordsVector(EmbeddedDocument):
    type=IntField()
    values=ListField(FloatField())

class DealW2v(Document):
    meta={'collection':'dealw2v'}
    primary=IntField(primary_key=True)
    words=ListField(StringField())
    vectorizedWords=DictField()