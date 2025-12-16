from typing import List, Set

HR_REQUIRED_COLUMNS: Set[str] = set()
SURVEY_REQUIRED_COLUMNS: Set[str] = set()


def required_hr_columns() -> List[str]:
    return sorted(HR_REQUIRED_COLUMNS)


def required_survey_columns() -> List[str]:
    return sorted(SURVEY_REQUIRED_COLUMNS)
