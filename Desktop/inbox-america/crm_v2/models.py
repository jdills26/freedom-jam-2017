from app import db

class top_by_cat2(db.Model):
    __tablename__ = "top_by_cat2"

    CAT1 = db.Column(db.String)
    index=db.Column(db.String, primary_key=True)
    modele_intitule= db.Column(db.String)
    sous_famille_intitule= db.Column(db.String)
    famille_intitule= db.Column(db.String)
    model_id= db.Column(db.String)

    def __repr__(self):
        return "{}".format(self.name)

class purchases(db.Model):
    __tablename__ = "inbox_table"

    client = db.Column(db.String)
    commande = db.Column(db.String, primary_key=True)
    date_commande_client = db.Column(db.String)
    code_article = db.Column(db.String)
    famille_intitule = db.Column(db.String)
    sous_famille_intitule = db.Column(db.String)
    modele_intitule = db.Column(db.String)
    category1 = db.Column(db.String)
    category2 = db.Column(db.String)
    model_id = db.Column(db.String)
    CAT1 = db.Column(db.String)
    #db.Integer

    def __repr__(self):
        return "{}".format(self.name)

