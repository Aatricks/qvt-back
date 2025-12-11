from typing import List, Set

HR_REQUIRED_COLUMNS: Set[str] = {"year", "absenteeism_rate", "turnover_rate"}
SURVEY_REQUIRED_COLUMNS: Set[str] = {"respondent_id", "question_label", "response_value"}


def required_hr_columns() -> List[str]:
    return sorted(HR_REQUIRED_COLUMNS)


def required_survey_columns() -> List[str]:
    return sorted(SURVEY_REQUIRED_COLUMNS)
