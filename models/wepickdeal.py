from mongoengine import Document,StringField,IntField,ObjectIdField,ReferenceField
from models import DealW2v

class WepickDeal(Document):
    meta={'collection':'wepickdeals'}
    primary=StringField(primary_key=True)
    cnt=IntField()
    deal=ReferenceField(DealW2v)