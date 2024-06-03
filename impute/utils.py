import pickle
def get_variables_description():
    description = {
        'BIRTH_WEIGHT_NC': 'Birth weight in grams. Minimum value is 500 grams. Maximum value is 5000 grams.',
        'GEST_AGE_NC': 'Gestational age in weeks. Minimum value is 25 weeks. Maximum value is 50 weeks.',
        'MAT_AGE_NC': 'Maternal age in years. Minimum value is 15 years. Maximum value is 50 years.',
        'APGAR_1_NC': 'APGAR score at 1 minute. The APGAR score is a test performed on newborns to assess their health. A score of 0-3 is considered critically low, 4-6 low, and 7-10 normal.',
        'APGAR_2_NC': 'APGAR score at 5 minutes. The APGAR score is a test performed on newborns to assess their health. A score of 0-3 is considered critically low, 4-6 low, and 7-10 normal.',
        'TOT_BIRTH_NUM_NC': 'Total number of births to the mother',
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

def load_trace(path):
    with open(path, 'rb') as file:
        trace = pickle.load(file)
    return trace
def get_imputable_variables():
    return ['BIRTH_WEIGHT_NC', 'GEST_AGE_NC', 'MAT_SMOKING_NC', 'BREASTFEED_BIRTH_FLG_NC']
