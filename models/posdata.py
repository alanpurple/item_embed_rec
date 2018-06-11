from mongoengine import Document,StringField,IntField,ObjectIdField

class PosData(Document):
    meta={'collection':'posdata'}
    _id=ObjectIdField()
    DealId=IntField()
    UserId=IntField()
    WepickRank=IntField()
    TransDate=StringField()
    Label=IntField()