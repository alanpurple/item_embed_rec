from mongoengine import Document,StringField,IntField,ObjectIdField,ReferenceField
from models import DealW2v

class PosData(Document):
    meta={'collection':'posdata'}
    DealId=ReferenceField(DealW2v)
    UserId=IntField()
    WepickRank=IntField()
    TransDate=StringField()
    Label=IntField()