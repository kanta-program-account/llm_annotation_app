# schema.py
from typing import List, Literal, Annotated
from pydantic import BaseModel, Field

class DimensionalEmotion(BaseModel):
    valence: Annotated[int, Field(ge=1, le=7, description="1 = very negative, 7 = very positive")]
    arousal: Annotated[int, Field(ge=1, le=7, description="1 = calm, 7 = excited")]
    dominance: Annotated[int, Field(ge=1, le=7, description="1 = submissive, 7 = confident")]
    engagement: Annotated[int, Field(ge=1, le=7, description="1 = detached, 7 = highly involved")]

class PrimaryEmotion(BaseModel):
    label: Literal["Angry", "Sad", "Happy", "Surprise", "Neutral", "Fear", "Disgust", "Contempt"]
    intensity: Annotated[int, Field(ge=1, le=7)]

AllEmotionLabels = Literal[
    "Angry", "Sad", "Happy", "Amused", "Neutral", "Frustrated", "Depressed", "Surprise",
    "Concerned", "Disgust", "Disappointed", "Excited", "Confused", "Annoyed", "Fear", "Contempt"
]

class AllEmotions(BaseModel):
    label: AllEmotionLabels
    intensity: Annotated[int, Field(ge=1, le=7)]

class CategoricalEmotion(BaseModel):
    primary_emotion: PrimaryEmotion
    all_emotions: Annotated[List[AllEmotions], Field(min_length=1)]

class InteractionalFeatures(BaseModel):
    empathy: Literal[0, 1]
    politeness: Literal[0, 1]
    disagreement: Literal[0, 1]
    tension: Literal[0, 1]
    rapport: Literal[0, 1]

DialogueActLabels = Literal[
    "Statement", "Backchannel/Acknowledge", "Opinion", "Abandoned/Uninterpretable", "Agreement/Accept",
    "Appreciation", "Yes-No-Question", "Non-verbal", "Yes Answers", "Conventional-closing",
    "WH-Question", "No Answers", "Response Acknowledgment", "Hedge", "Declarative Yes-No-Question",
    "Other", "Backchannel-Question", "Quotation", "Summarize/Reformulate",
    "Affirmative Non-Yes Answers", "Action-Directive", "Collaborative Completion", "Repeat-Phrase",
    "Open-Question", "Rhetorical-Questions", "Hold Before Answer/Agreement", "Reject",
    "Negative Non-No Answers", "Signal-Non-Understanding", "Other Answers", "Conventional-opening",
    "Or-Clause", "Dispreferred Answers", "3rd-Party-Talk", "Offers/Options/Commits", "Self-Talk",
    "Downplayer", "Maybe/Accept-Part", "Tag-Question", "Declarative WH-Question", "Apology", "Thanking"
]

class AnnotationSegment(BaseModel):
    speaker: str
    start: float
    end: float
    transcript: str
    multi_spk: bool
    dimensional_emotion: DimensionalEmotion
    categorical_emotion: CategoricalEmotion
    interactional_features: InteractionalFeatures
    dialogue_act: DialogueActLabels
    reasoning: str
    confidence: Annotated[float, Field(ge=0, le=1)]

class FinalAnnotationResponse(BaseModel):
    annotations: List[AnnotationSegment]