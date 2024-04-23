from enum import Enum


def get_variables_description():
    description = {
        'BIRTH_WEIGHT_NC': 'Birth weight in grams. Minimum value is 500 grams. Maximum value is 5000 grams.',
        'GEST_AGE_NC': 'Gestational age in weeks. Minimum value is 25 weeks. Maximum value is 50 weeks.',
        'MAT_AGE_NC': 'Maternal age in years. Minimum value is 15 years. Maximum value is 50 years.',
        'APGAR_1_NC': 'APGAR score at 1 minute. The APGAR score is a test performed on newborns to assess their health. A score of 0-3 is considered critically low, 4-6 low, and 7-10 normal.',
        'APGAR_2_NC': 'APGAR score at 5 minutes. The APGAR score is a test performed on newborns to assess their health. A score of 0-3 is considered critically low, 4-6 low, and 7-10 normal.',
        # 'TOT_BIRTH_NUM_NC': 'Total number of births to the mother',
        'CHILD_SEX_NC': 'Sex of the child. Male or Female. Male is encoded as 0 and Female as 1',
        'MAT_SMOKING_NC': 'Whether the mother smoked during pregnancy or not. Non-smoker is encoded as 0 and smoker as 1',
        'CHILD_ETHNIC_GROUP_NC': '0 means any White Background, including Welsh, English, Scottish, Northern Irish, Irish, British. 1 means any other ethnicity.',
        'LABOUR_ONSET_NC': 'Onset of labour. 1 means spontaneous, 2 means elective caesarean section, 3 means surgical induction by amniotomy, 4 means medical induction through administration of agents.',
        'BREASTFEED_BIRTH_FLG_NC': 'Whether the child was breastfed at birth or not. Not breastfed is encoded as 0 and breastfed as 1',
        'BIRTH_ORDER_NC': 'Birth order of the child in the case of twins. First child is encoded as 1, second child as 2.',
        'PREV_STILLBIRTH_NC': 'Whether the mother had a previous stillbirth or not. No stillbirth is encoded as 0 and stillbirth as 1',
        'PREV_LIVEBIRTHS_NC': 'Number of previous live births',
        'BREASTFEED_8_WKS_FLG_NC': 'Whether the child was breastfed at 8 weeks or not. Not breastfed is encoded as 0 and breastfed as 1',
        'MAT_REGION_NC': 'Region of the mother. Encoded as the a representative number for the LSOA region',
    }
    return description


def get_imputable_variables():
    return ['BIRTH_WEIGHT_NC', 'GEST_AGE_NC', 'MAT_AGE_NC', 'MAT_SMOKING_NC', 'BREASTFEED_BIRTH_FLG_NC',
            'MAT_REGION_NC']


def create_induced_variables():
    induced_variables = []
    for variable in InducedVariableName.__members__:
        correlated_vars = get_correlated_variables(InducedVariableName[variable])
        induced_variable = InducedVariable(InducedVariableName[variable], correlated_vars)
        induced_variables.append(induced_variable)
    return induced_variables


def get_correlated_variables(variable):
    if variable == InducedVariableName.MS:
        return [InductorVariable.BW, InductorVariable.MA, InductorVariable.PS]
    elif variable == InducedVariableName.BB:
        return [InductorVariable.B8, InductorVariable.MS, InductorVariable.MA, InductorVariable.LO, InductorVariable.PL]
    elif variable == InducedVariableName.BW:
        return [InductorVariable.GA, InductorVariable.MS, InductorVariable.BT]
    elif variable == InducedVariableName.GA:
        return [InductorVariable.BW, InductorVariable.BT]


class InducedVariableName(Enum):
    BW = 'BIRTH_WEIGHT_NC'
    GA = 'GEST_AGE_NC'
    # MA = 'MAT_AGE_NC'
    # MR = 'MAT_REGION_NC'
    MS = 'MAT_SMOKING_NC'
    BB = 'BREASTFEED_BIRTH_FLG_NC'


class InducedVariable:
    def __init__(self, variable: InducedVariableName, correlated_variables):
        self.name = variable.value
        self.correlated_variables = [correlated_variable.value for correlated_variable in correlated_variables]


class InductorVariable(Enum):
    BW = 'BIRTH_WEIGHT_NC'
    GA = 'GEST_AGE_NC'
    MA = 'MAT_AGE_NC'
    MS = 'MAT_SMOKING_NC'
    BB = 'BREASTFEED_BIRTH_FLG_NC'
    MR = 'MAT_REGION_NC'
    BT = 'BIRTH_ORDER_NC'
    PS = 'PREV_STILLBIRTH_NC'
    EG = 'CHILD_ETHNIC_GROUP_NC'
    SX = 'CHILD_SEX_NC'
    LO = 'LABOUR_ONSET_NC'
    AP1 = 'APGAR_1_NC'
    AP2 = 'APGAR_2_NC'
    TB = 'TOT_BIRTH_NUM_NC'
    PL = 'PREV_LIVEBIRTHS_NC'
    B8 = 'BREASTFEED_8_WKS_FLG_NC'


class Correlation:
    def __init__(self, inductor, induced, results, description):
        self.inductor = inductor
        self.induced = induced
        self.description = description
        self.results = results
