from mongoengine import Document,StringField,IntField,ObjectIdField,ReferenceField
from models import DealW2v

class WepickDeal(Document):
    _id=StringField()
    cnt=IntField()
    deal:ReferenceField(DealW2v)
