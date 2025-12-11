from typing import List, Set

HR_REQUIRED_COLUMNS: Set[str] = {
    "ID",
    "Sexe",
    "Age",
    "Contrat",
    "Temps",
    "Encadre",
    "Ancienne",
    "Secteur",
    "TailleOr",
}
SURVEY_REQUIRED_COLUMNS: Set[str] = set()


def required_hr_columns() -> List[str]:
    return sorted(HR_REQUIRED_COLUMNS)


def required_survey_columns() -> List[str]:
    return sorted(SURVEY_REQUIRED_COLUMNS)
