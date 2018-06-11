from mongoengine import Document,StringField,IntField,FloatField,ObjectIdField,ListField,EmbeddedDocumentField,ReferenceField,EmbeddedDocument

class WordsVector(EmbeddedDocument):
    type=IntField()
    values=ListField(FloatField())

class DealW2v(Document):
    meta={'collection':'dealw2v'}
    _id=ObjectIdField()
    v=IntField(unique=True)
    words=ListField(StringField())
    vectorizedWords=EmbeddedDocumentField(WordsVector)