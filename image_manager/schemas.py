from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, TypedDict, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# =========================================================
# Model Enum
# =========================================================

class ImageModel(str, Enum):
    OPENAI_DALLE3 = "openai_dalle3"
    STABLE_DIFFUSION = "stable_diffusion"
    MIDJOURNEY = "midjourney"
    GOOGLE_IMAGEN = "google_imagen"
    GROK_IMAGINE_IMAGE = "grok_imagine_image"


# =========================================================
# OpenAI DALL-E 3
# =========================================================

class Dalle3Style(str, Enum):
    VIVID = "vivid"
    NATURAL = "natural"


class Dalle3Quality(str, Enum):
    STANDARD = "standard"
    HD = "hd"


class Dalle3Size(str, Enum):
    SQUARE = "1024x1024"
    PORTRAIT = "1024x1792"
    LANDSCAPE = "1792x1024"


class Dalle3ResponseFormat(str, Enum):
    URL = "url"
    B64_JSON = "b64_json"


class OpenAIDalle3Kwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style: Optional[Dalle3Style] = Field(default=None, description="이미지 스타일 강도")
    quality: Optional[Dalle3Quality] = Field(default=None, description="생성 품질")
    size: Optional[Dalle3Size] = Field(default=None, description="출력 이미지 크기")
    response_format: Optional[Dalle3ResponseFormat] = Field(default=None, description="응답 포맷")
    user: Optional[str] = Field(default=None, description="사용자 식별자")


# =========================================================
# Stable Diffusion
# =========================================================

class StableDiffusionKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    negative_prompt: Optional[str] = Field(default=None, description="제외하고 싶은 요소")
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, description="프롬프트 반영 강도")
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=300, description="추론 스텝 수")
    width: Optional[int] = Field(default=None, ge=64, description="이미지 너비")
    height: Optional[int] = Field(default=None, ge=64, description="이미지 높이")
    seed: Optional[int] = Field(default=None, ge=0, description="재현 가능한 생성용 시드")
    scheduler: Optional[str] = Field(default=None, description="샘플링 스케줄러")
    num_images: Optional[int] = Field(default=None, ge=1, le=16, description="생성 이미지 개수")
    clip_skip: Optional[int] = Field(default=None, ge=1, le=12, description="CLIP 레이어 스킵 설정")
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="img2img/inpaint 시 원본 반영 비율")

    @field_validator("width", "height")
    @classmethod
    def validate_multiple_of_8(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v % 8 != 0:
            raise ValueError("width/height 는 보통 8의 배수여야 합니다.")
        return v


# =========================================================
# Midjourney
# =========================================================

class MidjourneyQuality(str, Enum):
    Q_025 = ".25"
    Q_05 = ".5"
    Q_1 = "1"
    Q_2 = "2"


class MidjourneyKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspect_ratio: Optional[str] = Field(default=None, description='가로세로 비율 예: "1:1", "16:9"')
    stylize: Optional[int] = Field(default=None, ge=0, le=1000, description="스타일 적용 강도")
    chaos: Optional[int] = Field(default=None, ge=0, le=100, description="결과 다양성")
    quality: Optional[MidjourneyQuality] = Field(default=None, description="렌더 품질")
    seed: Optional[int] = Field(default=None, ge=0, description="재현용 시드")
    weird: Optional[int] = Field(default=None, ge=0, le=3000, description="독특함 강도")
    tile: Optional[bool] = Field(default=None, description="반복 패턴 생성 여부")
    version: Optional[str] = Field(default=None, description='모델 버전 예: "6", "niji"')
    stop: Optional[int] = Field(default=None, ge=10, le=100, description="렌더링 조기 종료 비율")

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: Optional[str]) -> Optional[str]:
        return validate_ratio_string(v)


# =========================================================
# Google Imagen / Gemini Imagen
# =========================================================

class GoogleImagenImageSize(str, Enum):
    K1 = "1K"
    K2 = "2K"


class GoogleImagenAspectRatio(str, Enum):
    RATIO_1_1 = "1:1"
    RATIO_3_4 = "3:4"
    RATIO_4_3 = "4:3"
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"


class GoogleImagenPersonGeneration(str, Enum):
    ALLOW_ALL = "allow_all"
    ALLOW_ADULT = "allow_adult"
    DONT_ALLOW = "dont_allow"


class GoogleImagenSafetyFilterLevel(str, Enum):
    BLOCK_LOW_AND_ABOVE = "block_low_and_above"
    BLOCK_MEDIUM_AND_ABOVE = "block_medium_and_above"
    BLOCK_ONLY_HIGH = "block_only_high"
    BLOCK_NONE = "block_none"


class GoogleImagenKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number_of_images: Optional[int] = Field(
        default=None,
        ge=1,
        le=4,
        description="생성 이미지 개수 (Gemini Imagen 문서 기준 1~4)"
    )
    image_size: Optional[GoogleImagenImageSize] = Field(
        default=None,
        description='출력 크기 예: "1K", "2K"'
    )
    aspect_ratio: Optional[GoogleImagenAspectRatio] = Field(
        default=None,
        description='가로세로 비율 예: "1:1", "16:9"'
    )

    add_watermark: Optional[bool] = Field(
        default=None,
        description="SynthID 워터마크 적용 여부"
    )
    person_generation: Optional[GoogleImagenPersonGeneration] = Field(
        default=None,
        description="사람/얼굴 생성 허용 수준"
    )
    safety_filter_level: Optional[GoogleImagenSafetyFilterLevel] = Field(
        default=None,
        description="안전 필터 강도"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="결정적 결과용 시드"
    )
    enhance_prompt: Optional[bool] = Field(
        default=None,
        description="프롬프트 향상 사용 여부"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="제외할 요소. 일부 Imagen 3 모델에서만 지원될 수 있음"
    )


# =========================================================
# xAI Grok Imagine Image
# =========================================================

class GrokOutputFormat(str, Enum):
    URL = "url"
    B64_JSON = "b64_json"


class GrokAspectRatio(str, Enum):
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_3_2 = "3:2"
    RATIO_2_3 = "2:3"


class GrokResolution(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GrokImagineImageKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="동일 프롬프트 기반 생성 이미지 개수"
    )
    aspect_ratio: Optional[GrokAspectRatio] = Field(
        default=None,
        description='가로세로 비율 예: "1:1", "16:9"'
    )
    resolution: Optional[GrokResolution] = Field(
        default=None,
        description="출력 해상도 레벨"
    )
    response_format: Optional[GrokOutputFormat] = Field(
        default=None,
        description="응답 포맷"
    )

    image_url: Optional[str] = Field(
        default=None,
        description="편집용 원본 이미지 URL 또는 data URI"
    )


# =========================================================
# TypedDict
# =========================================================

class OpenAIDalle3KwargsDict(TypedDict, total=False):
    style: Literal["vivid", "natural"]
    quality: Literal["standard", "hd"]
    size: Literal["1024x1024", "1024x1792", "1792x1024"]
    response_format: Literal["url", "b64_json"]
    user: str


class StableDiffusionKwargsDict(TypedDict, total=False):
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    width: int
    height: int
    seed: int
    scheduler: str
    num_images: int
    clip_skip: int
    strength: float


class MidjourneyKwargsDict(TypedDict, total=False):
    aspect_ratio: str
    stylize: int
    chaos: int
    quality: Literal[".25", ".5", "1", "2"]
    seed: int
    weird: int
    tile: bool
    version: str
    stop: int


class GoogleImagenKwargsDict(TypedDict, total=False):
    number_of_images: int
    image_size: Literal["1K", "2K"]
    aspect_ratio: Literal["1:1", "3:4", "4:3", "9:16", "16:9"]
    add_watermark: bool
    person_generation: Literal["allow_all", "allow_adult", "dont_allow"]
    safety_filter_level: Literal[
        "block_low_and_above",
        "block_medium_and_above",
        "block_only_high",
        "block_none",
    ]
    seed: int
    enhance_prompt: bool
    negative_prompt: str


class GrokImagineImageKwargsDict(TypedDict, total=False):
    n: int
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
    resolution: Literal["low", "medium", "high"]
    response_format: Literal["url", "b64_json"]
    image_url: str


# =========================================================
# Union Type for DTOs
# =========================================================

ModelKwargsType = Union[
    OpenAIDalle3Kwargs,
    StableDiffusionKwargs,
    MidjourneyKwargs,
    GoogleImagenKwargs,
    GrokImagineImageKwargs,
]


# =========================================================
# Validation Helpers
# =========================================================

MODEL_KWARGS_SCHEMA_MAP: dict[ImageModel, type[BaseModel]] = {
    ImageModel.OPENAI_DALLE3: OpenAIDalle3Kwargs,
    ImageModel.STABLE_DIFFUSION: StableDiffusionKwargs,
    ImageModel.MIDJOURNEY: MidjourneyKwargs,
    ImageModel.GOOGLE_IMAGEN: GoogleImagenKwargs,
    ImageModel.GROK_IMAGINE_IMAGE: GrokImagineImageKwargs,
}


class KwargsValidationError(ValueError):
    """모델별 kwargs 검증 실패용 예외"""


def validate_ratio_string(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v

    parts = v.split(":")
    if len(parts) != 2:
        raise ValueError('aspect_ratio 형식은 "W:H" 이어야 합니다. 예: "16:9"')

    try:
        w = int(parts[0])
        h = int(parts[1])
    except ValueError as exc:
        raise ValueError('aspect_ratio 는 정수 비율이어야 합니다. 예: "4:3"') from exc

    if w <= 0 or h <= 0:
        raise ValueError("aspect_ratio 의 각 값은 1 이상의 정수여야 합니다.")
    return v


def format_pydantic_error(model_name: str, exc: ValidationError) -> str:
    lines: list[str] = [f"{model_name} kwargs 검증 실패:"]
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "알 수 없는 오류")
        input_value = err.get("input", None)
        lines.append(f"- {loc}: {msg} (입력값: {input_value!r})")
    return "\n".join(lines)


def validate_model_kwargs(model_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        model_enum = ImageModel(model_name)
    except ValueError as exc:
        supported = ", ".join(m.value for m in ImageModel)
        raise KwargsValidationError(
            f"지원하지 않는 model_name 입니다: {model_name!r}. 지원 모델: {supported}"
        ) from exc

    schema = MODEL_KWARGS_SCHEMA_MAP[model_enum]

    try:
        validated = schema.model_validate(kwargs)
    except ValidationError as exc:
        raise KwargsValidationError(format_pydantic_error(model_name, exc)) from exc

    return validated.model_dump(exclude_none=True, mode="json")


def get_allowed_properties(model_name: str) -> dict[str, dict[str, Any]]:
    try:
        model_enum = ImageModel(model_name)
    except ValueError as exc:
        supported = ", ".join(m.value for m in ImageModel)
        raise ValueError(
            f"지원하지 않는 model_name 입니다: {model_name!r}. 지원 모델: {supported}"
        ) from exc

    schema = MODEL_KWARGS_SCHEMA_MAP[model_enum]
    result: dict[str, dict[str, Any]] = {}

    for name, field in schema.model_fields.items():
        result[name] = {
            "annotation": str(field.annotation),
            "required": field.is_required(),
            "default": field.default,
            "description": field.description,
        }
    return result

# =========================================================
# Request Model
# =========================================================

class ImageGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, description="이미지 생성 프롬프트")
    model_name: ImageModel = Field(..., description="이미지 생성 모델")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="모델별 전용 옵션")

    @model_validator(mode="after")
    def validate_kwargs_by_model(self) -> "ImageGenerateRequest":
        self.kwargs = validate_model_kwargs(self.model_name.value, self.kwargs)
        return self
