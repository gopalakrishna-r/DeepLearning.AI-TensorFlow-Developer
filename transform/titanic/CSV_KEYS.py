class titanic:
    TEXT_FEAT_KEYS = ["Name", "Cabin", "Ticket"]

    NUMERICAL_CAT_NONCONT_KEYS = ["Parch", "PassengerId", "Pclass", "SibSp"]

    NUMERICAL_CAT_CONT_KEYS = ["Age", "Fare"]

    CATEGORICAL_FEAT_KEYS = ["Sex", "Embarked"]

    TARGET_FEATURE_NAME = "Survived"
    TARGET_FEATURE_LABELS = [0, 1]
