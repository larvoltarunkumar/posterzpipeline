from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Literal


# ============================================================
# POOLED POPULATION SCHEMAS
# ============================================================

ArmType = Literal[
    "Experimental",
    "Comparator",
    "Single-arm",
    "External control",
    "Dose level",
    "Cohort",
    "Other"
]

PopulationType = Literal[
    "Overall",
    "Analysis set",
    "Cohort",
    "Baseline characteristic",
    "Subgroup",
    "Other"
]

IntegratedType = Literal[
    "Integrated population",
    "Pooled analysis",
    "Other"
]


class DesignSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Optional[str] = None


class TrialPopulationDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Optional[str] = None


class TrialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_key: str
    trial_id_list: List[str] = Field(default_factory=list)
    trial_label: Optional[str] = None
    phase: Optional[str] = None
    study_name: Optional[str] = None
    allocation: Optional[str] = None
    design_summary: DesignSummary
    trial_population_details: TrialPopulationDetails
    overall_N: Optional[str] = None


class ArmRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm_key: str
    arm_name: Optional[str] = None
    arm_type: Optional[ArmType] = None
    treatment_description: Optional[str] = None
    dose_schedule: Optional[str] = None


class PopulationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population_key: str
    population_type: PopulationType
    parent: Optional[str] = None
    child: Optional[str] = None
    population_description: Optional[str] = None
    N: Optional[str] = None


class TrialArmLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_key: str
    linked_arm_keys: List[str] = Field(default_factory=list)


class TrialPopulationLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_key: str
    linked_population_keys: List[str] = Field(default_factory=list)
    linked_arm_keys: List[str] = Field(default_factory=list)


class IntegratedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    integrated_key: str
    integrated_type: IntegratedType
    source_trial_keys: List[str] = Field(default_factory=list)
    population_description: Optional[str] = None
    N: Optional[str] = None
    linked_population_keys: List[str] = Field(default_factory=list)
    linked_arm_keys: List[str] = Field(default_factory=list)


class MultiTrialExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_records: List[TrialRecord] = Field(default_factory=list)
    arm_records: List[ArmRecord] = Field(default_factory=list)
    population_records: List[PopulationRecord] = Field(default_factory=list)
    trial_arm_links: List[TrialArmLink] = Field(default_factory=list)
    trial_population_links: List[TrialPopulationLink] = Field(default_factory=list)
    integrated_records: List[IntegratedRecord] = Field(default_factory=list)


# ============================================================
# KM SURVIVAL SCHEMAS
# ============================================================

TimeUnit = Literal["months", "years", "weeks", "days"]


class TrialMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_id: Optional[str] = None
    phase: Optional[str] = None
    study_name: Optional[str] = None


class ArmLevelSurvivalOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    survival_outcome_id: int
    trial_id: Optional[str] = None
    trial_label: Optional[str] = None
    arm_description: Optional[str] = None

    population_type: PopulationType
    population_description: Optional[str] = None

    endpoint_description: Optional[str] = None
    endpoint_name: Optional[str] = None
    endpoint_label: Optional[str] = None

    assessment_type: Optional[str] = None
    review_board: Optional[str] = None
    review_criteria: Optional[str] = None
    other_details: Optional[str] = None

    arm_n: Optional[int] = None
    median_survival: Optional[str] = None
    survival_rate: Optional[str] = None
    events_n: Optional[int] = None
    assessment_denominator_n: Optional[int] = None

    p_value: Optional[float] = None
    time_unit: Optional[TimeUnit] = None

    @field_validator("survival_outcome_id")
    @classmethod
    def survival_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("survival_outcome_id must be >= 1")
        return v

    @field_validator("arm_n", "events_n", "assessment_denominator_n")
    @classmethod
    def non_negative_ints(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 0:
            raise ValueError("Count fields must be >= 0")
        return v

    @field_validator("p_value")
    @classmethod
    def p_value_range(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < 0 or v > 1:
            raise ValueError("p_value must be between 0 and 1")
        return v


class SurvivalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_metadata: TrialMetadata = Field(default_factory=TrialMetadata)
    arm_level_survival_outcomes: List[ArmLevelSurvivalOutcome] = Field(default_factory=list)


# ============================================================
# BASELINE CHARACTERISTICS SCHEMAS
# ============================================================

BaselineParent = Literal["Overall", "Cohort", "Subgroup", "Other", None]


class BaselineCharacteristic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_id: int = Field(..., description="Sequential ID starting from 1")

    trial_id: Optional[str] = None
    trial_label: Optional[str] = None

    arm_key: Optional[str] = None
    arm_description: Optional[str] = None

    population_key: Optional[str] = None
    population_type: PopulationType
    population_description: Optional[str] = None

    baseline_parent: BaselineParent = None
    parent_description: Optional[str] = None

    baseline_category_label: Optional[str] = None
    group_label: Optional[str] = None
    group_text: Optional[str] = None

    measure: Optional[str] = None
    measure_value: Optional[str] = None

    population_n: Optional[int] = None
    population_percentage: Optional[float] = None

    @field_validator("baseline_id")
    @classmethod
    def baseline_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("baseline_id must be >= 1")
        return v

    @field_validator("population_n")
    @classmethod
    def non_negative_n(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 0:
            raise ValueError("population_n must be >= 0")
        return v

    @field_validator("population_percentage")
    @classmethod
    def percent_range(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < 0 or v > 100:
            raise ValueError("population_percentage must be between 0 and 100")
        return v


class BaselineOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bc_types: List[BaselineCharacteristic] = Field(default_factory=list)


# ============================================================
# RESPONSE OUTCOMES SCHEMAS
# ============================================================

ResponseMetricClass = Literal["rate", "duration", "time_to_response"]


class ResultObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: Optional[int] = None
    percentage: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    p_value: Optional[float] = None
    odds_ratio: Optional[float] = None

    median: Optional[float] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    duration_unit: Optional[TimeUnit] = None


class ArmLevelResponseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_outcome_id: int

    trial_id: Optional[str] = None
    trial_label: Optional[str] = None

    arm_description: Optional[str] = None

    population_type: PopulationType
    population_description: Optional[str] = None

    assessment_type: Optional[str] = None
    review_board: Optional[str] = None
    review_criteria: Optional[str] = None
    other_details: Optional[str] = None

    arm_n: Optional[int] = None
    assessment_denominator_n: Optional[int] = None

    response_type_name: Optional[str] = None
    response_metric_class: Optional[ResponseMetricClass] = None

    result: Optional[ResultObject] = None

    @field_validator("response_outcome_id")
    @classmethod
    def id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("response_outcome_id must be >= 1")
        return v

    @field_validator("arm_n", "assessment_denominator_n")
    @classmethod
    def non_negative_ints(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 0:
            raise ValueError("Count fields must be >= 0")
        return v


class ResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_metadata: TrialMetadata = Field(default_factory=TrialMetadata)
    arm_level_response_outcomes: List[ArmLevelResponseOutcome] = Field(default_factory=list)
