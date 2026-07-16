"""
Schema Configuration Module
Defines annotation schemas and field mappings for IAA evaluation
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class FieldConfig:
    """Configuration for a single annotation field"""
    name: str
    field_type: str  # 'categorical' or 'text'
    separator: Optional[str] = None  # delimiter for multi-value fields: ',' or '\n'
    metrics: List[str] = None  # List of metrics to calculate for this field

    def __post_init__(self):
        if self.metrics is None:
            # Default metrics based on field type
            if self.field_type == 'categorical':
                self.metrics = ['cohen_kappa', 'observed_agreement', 'pabak', 'gwet_ac']
            elif self.field_type == 'text':
                self.metrics = ['bert', 'jaccard', 'meteor']


class SchemaConfig:
    """Schema configuration manager"""

    def __init__(
        self,
        categorical_fields: List[str],
        text_fields: List[str],
        field_separators: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize schema configuration

        Args:
            categorical_fields: List of categorical field names
            text_fields: List of text similarity field names
            field_separators: Optional mapping of field name → delimiter character
                              e.g. {'alternativeKeywords': ',', 'goodEvidence': '\n'}
        """
        self.categorical_fields = categorical_fields
        self.text_fields = text_fields
        self._field_configs = {}
        separators = field_separators or {}

        # Create FieldConfig objects for all fields
        for field in categorical_fields:
            self._field_configs[field] = FieldConfig(name=field, field_type='categorical')

        for field in text_fields:
            sep = separators.get(field)
            self._field_configs[field] = FieldConfig(
                name=field, field_type='text', separator=sep
            )

    def get_categorical_fields(self) -> List[str]:
        """Get list of categorical field names"""
        return self.categorical_fields

    def get_text_fields(self) -> List[str]:
        """Get list of text similarity field names"""
        return self.text_fields

    def get_all_fields(self) -> List[str]:
        """Get list of all field names"""
        return self.categorical_fields + self.text_fields

    def get_field_config(self, field_name: str) -> Optional[FieldConfig]:
        """Get configuration for a specific field"""
        return self._field_configs.get(field_name)

    def is_categorical(self, field_name: str) -> bool:
        """Check if a field is categorical"""
        return field_name in self.categorical_fields

    def is_text(self, field_name: str) -> bool:
        """Check if a field is text similarity"""
        return field_name in self.text_fields

    def validate_dataframe(self, df, missing_ok: bool = True) -> Dict[str, List[str]]:
        """
        Validate that dataframe has required fields

        Args:
            df: pandas DataFrame to validate
            missing_ok: If True, missing fields are warnings not errors

        Returns:
            dict with 'present' and 'missing' field lists
        """
        all_fields = set(self.get_all_fields())
        df_fields = set(df.columns)

        present = list(all_fields.intersection(df_fields))
        missing = list(all_fields - df_fields)

        return {
            'present': present,
            'missing': missing,
            'extra': list(df_fields - all_fields)
        }


def create_field_annotating_schema() -> SchemaConfig:
    """
    Create schema for Field_Annotating_Evaluation.txt

    Returns:
        SchemaConfig: Configured schema with 22 categorical + 30 text fields
    """

    # 22 Categorical fields from Field_Annotating_Evaluation.txt
    categorical_fields = [
        # Interpretation phase
        'targetExplanation',        # dropdown: repeated-emphasis, explicitly-stated, etc.
        'stanceTarget',              # dropdown: favor/against/neutral
        'knowledgeDomain',           # dropdown: technical/legal/cultural/etc.

        # Analysis phase
        'reasoningStructure',        # dropdown: specific-cases, outcomes, etc.
        'span1',                     # dropdown: core-claim/minor-claim/premise
        'span2',                     # dropdown: core-claim/minor-claim/premise
        'span3',                     # dropdown: core-claim/minor-claim/premise
        'reasoningType',             # dropdown: inductive/deductive/none
        'hasAssumption',             # dropdown: yes/no

        # Inference phase
        'inferenceStrength',         # rating: 1-3

        # Evaluation phase
        'credibilityFactors',        # checkboxes: bias/hearsay/emotional/etc.
        'logicalFallacy',            # dropdown: false-dilemma/hasty-gen/etc.
        'trustworthiness',           # checkboxes: evidence/logical/common/cannot

        # Inference (continued)
        'primaryDomain',             # dropdown: interpersonal/personal/social/etc.
        'alternativeType',           # dropdown: applies-when/other-factors/misunderstands

        # Self-regulation phase
        'biasDetection',             # checkboxes: fear-anxiety/first-impression/etc.
        'heuristicDetection',        # checkboxes: gut-feeling/familiar-pattern/etc.
        'changeDecision',            # dropdown: yes/no
        'revisionPhases',            # checkboxes: interpretation/analysis/etc.
        'revisionType'               # dropdown: minor-adjustment/major-change/etc.
    ]

    # 30 Text similarity fields from Field_Annotating_Evaluation.txt
    text_fields = [
        # Interpretation phase
        'initialUnderstandingTarget',  # free text: "The argument talks about..."
        'paraphrasingUnderstanding',   # free text paraphrase

        # Analysis phase
        'coreClaim',                   # span selection/paraphrase
        'minorClaim',                  # span selection/paraphrase
        'premise',                     # span selection/paraphrase
        'deductiveTermX',              # free text terms
        'deductiveTermY',
        'inductiveTermX',
        'inductiveTermY',
        'missingComponent',            # keywords for missing components

        # Inference phase
        'positiveConsequences',        # keywords/phrases
        'negativeConsequences',        # keywords/phrases
        'alternativeKeywords',         # 3 keywords max

        # Evaluation phase
        'goodEvidence',                # keywords/phrases
        'badEvidence',                 # keywords/phrases
        'trustworthinessExplanation',  # explanation text
        'noFallacyExplanation',        # explanation text
        'fallacySpan1',                # span identification
        'fallacySpan2',                # span identification

        # Self-regulation phase
        'errorDetection',              # free text error description
        'revisionReason',              # free text explanation

        # Socratic questions (generated after all phases)
        'socraticQuestion1',           # generated question
        'socraticQuestion2',           # generated question
        'socraticQuestion3'            # generated question
    ]

    # Fields whose values contain multiple items separated by a delimiter
    field_separators = {
        'alternativeKeywords': ',',    # e.g. "individuality, self-expression"
        'positiveConsequences': ',',   # e.g. "reduce bullying, reduce time preparing"
        'negativeConsequences': ',',   # e.g. "lack of individuality, lack of expression"
        'goodEvidence': '\n',          # newline-separated evidence sentences
        'badEvidence': '\n',           # newline-separated evidence sentences
    }

    return SchemaConfig(
        categorical_fields=categorical_fields,
        text_fields=text_fields,
        field_separators=field_separators,
    )


def create_kim_paul_schema() -> SchemaConfig:
    """
    Create schema for existing Kim/Paul annotation structure
    (For backwards compatibility and regression testing)

    Returns:
        SchemaConfig: Configured schema matching IAA_test.py structure
    """

    categorical_fields = [
        'stanceTarget',
        'knowledgeDomain',
        'hasAssumption',
        'reasoningType',
        'consequence',
        'primaryDomain',
        'alternativeType',
        'inferenceStrength',
        'credibilityFactors',
        'logicalFallacy',
        'trustworthiness'
    ]

    text_fields = [
        'initialUnderstandingTarget',
        'paraphrasingUnderstanding',
        'coreClaim',
        'minorClaim',
        'premise',
        'missingComponent',
        'alternativeKeywords',
        'trustExplanation'
    ]

    return SchemaConfig(
        categorical_fields=categorical_fields,
        text_fields=text_fields
    )
