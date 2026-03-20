"""
Batch variants of the built-in Nano Banana / Nano Banana 2 API nodes.

Fires `batch_size` concurrent requests via asyncio.gather so total wall-clock
time is roughly equal to a single generation rather than batch_size × that.
All billing goes through the same ComfyUI proxy and credits as the originals.
"""

import asyncio
import base64
from io import BytesIO

import torch
import torch.nn.functional as F

from comfy_api.latest import IO, Input
from comfy_api_nodes.apis.gemini import (
    GeminiContent,
    GeminiGenerateContentResponse,
    GeminiImageConfig,
    GeminiImageGenerateContentRequest,
    GeminiImageGenerationConfig,
    GeminiPart,
    GeminiRole,
    GeminiSystemInstructionContent,
    GeminiTextPart,
    GeminiThinkingConfig,
)
from comfy_api_nodes.nodes_gemini import (
    GEMINI_IMAGE_SYS_PROMPT,
    GEMINI_IMAGE_2_PRICE_BADGE,
    GeminiImageModel,
    create_image_parts,
    get_parts_by_type,
    get_text_from_response,
    calculate_tokens_price,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    download_url_to_image_tensor,
    get_number_of_images,
    sync_op,
    validate_string,
)


def _resize_to_match(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Resize all BHWC tensors to match the largest one in the list (bilinear)."""
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    out = []
    for t in tensors:
        if t.shape[1] != max_h or t.shape[2] != max_w:
            t = F.interpolate(
                t.permute(0, 3, 1, 2),  # BHWC -> BCHW
                size=(max_h, max_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)  # BCHW -> BHWC
        out.append(t)
    return out


async def _get_image_from_response(response: GeminiGenerateContentResponse) -> torch.Tensor:
    """Like the core get_image_from_response but, when parts have different sizes
    (e.g. a 1K base + 2K upscale in the same response), keeps only the largest."""
    parts = get_parts_by_type(response, "image/*")
    image_tensors = []
    for part in parts:
        if part.inlineData:
            image_data = base64.b64decode(part.inlineData.data)
            img = bytesio_to_image_tensor(BytesIO(image_data))
        else:
            img = await download_url_to_image_tensor(part.fileData.fileUri)
        image_tensors.append(img)
    if not image_tensors:
        return torch.zeros((1, 1024, 1024, 4))
    if len(image_tensors) == 1:
        return image_tensors[0]
    # If parts differ in size, keep only the highest-resolution one.
    sizes = [t.shape[1] * t.shape[2] for t in image_tensors]
    if len(set(sizes)) > 1:
        return image_tensors[sizes.index(max(sizes))]
    return torch.cat(image_tensors, dim=0)


async def _single_call(cls, model: str, parts: list[GeminiPart], gen_config: GeminiImageGenerationConfig, system_prompt_obj) -> GeminiGenerateContentResponse:
    """Fire one Gemini image generation request."""
    return await sync_op(
        cls,
        ApiEndpoint(path=f"/proxy/vertexai/gemini/{model}", method="POST"),
        data=GeminiImageGenerateContentRequest(
            contents=[GeminiContent(role=GeminiRole.user, parts=parts)],
            generationConfig=gen_config,
            systemInstruction=system_prompt_obj,
        ),
        response_model=GeminiGenerateContentResponse,
        price_extractor=calculate_tokens_price,
        monitor_progress=False,
    )


class NanoBananaBatch(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NanoBananaBatch",
            display_name="Nano Banana Batch",
            category="api node/image/Gemini",
            description=(
                "Generate a batch of images with Nano Banana (Google Gemini Image) in a single run. "
                "All batch_size requests are fired concurrently so total time ≈ one image. "
                "Cost = batch_size × per-image price."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt for generation",
                    default="",
                ),
                IO.Int.Input(
                    "batch_size",
                    default=4,
                    min=1,
                    max=8,
                    tooltip="Number of images to generate concurrently per run. If the API returns varying resolutions across calls, all images are resized to match the first result.",
                ),
                IO.Combo.Input(
                    "model",
                    options=GeminiImageModel,
                    default=GeminiImageModel.gemini_2_5_flash_image,
                    tooltip="The Gemini model to use.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Base seed. Each image in the batch uses seed+i for variety.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference image(s). Uploaded once and shared across all batch calls.",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="auto",
                    optional=True,
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    optional=True,
                    advanced=True,
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.039,"format":{"suffix":"/Image (1K) × batch_size","approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        batch_size: int,
        model: str,
        seed: int,
        images: Input.Image | None = None,
        files: list[GeminiPart] | None = None,
        aspect_ratio: str = "auto",
        response_modalities: str = "IMAGE+TEXT",
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)

        # Build parts once — images are uploaded here and the resulting URL
        # references are safe to share across all concurrent requests.
        base_parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images is not None:
            base_parts.extend(await create_image_parts(cls, images))
        if files is not None:
            base_parts.extend(files)

        image_config = GeminiImageConfig() if aspect_ratio == "auto" else GeminiImageConfig(aspectRatio=aspect_ratio)
        gen_config = GeminiImageGenerationConfig(
            responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
            imageConfig=image_config,
        )

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(
                parts=[GeminiTextPart(text=system_prompt)], role=None
            )

        responses = await asyncio.gather(*[
            _single_call(cls, model, base_parts, gen_config, gemini_system_prompt)
            for _ in range(batch_size)
        ])

        image_tensors = []
        texts = []
        for response in responses:
            image_tensors.append(await _get_image_from_response(response))
            texts.append(get_text_from_response(response))

        return IO.NodeOutput(torch.cat(_resize_to_match(image_tensors), dim=0), "\n---\n".join(texts))


class NanoBanana2Batch(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NanoBanana2Batch",
            display_name="Nano Banana 2 Batch",
            category="api node/image/Gemini",
            description=(
                "Generate a batch of images with Nano Banana 2 (Google Gemini Image) in a single run. "
                "All batch_size requests are fired concurrently so total time ≈ one image. "
                "Cost = batch_size × per-image price."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt describing the image to generate or edits to apply.",
                    default="",
                ),
                IO.Int.Input(
                    "batch_size",
                    default=4,
                    min=1,
                    max=8,
                    tooltip="Number of images to generate concurrently per run. If the API returns varying resolutions across calls, all images are resized to match the first result.",
                ),
                IO.Combo.Input(
                    "model",
                    options=["gemini-3-pro-image-preview", "Nano Banana 2 (Gemini 3.1 Flash Image)"],
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Base seed. Each image in the batch uses seed+i for variety.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="auto",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "2K", "4K"],
                    tooltip="Target output resolution. For 2K/4K the native Gemini upscaler is used.",
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    advanced=True,
                ),
                IO.Combo.Input(
                    "thinking_level",
                    options=["MINIMAL", "HIGH"],
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference image(s). Uploaded once and shared across all batch calls.",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=GEMINI_IMAGE_2_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        batch_size: int,
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        thinking_level: str,
        images: Input.Image | None = None,
        files: list[GeminiPart] | None = None,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)

        if model == "Nano Banana 2 (Gemini 3.1 Flash Image)":
            model = "gemini-3.1-flash-image-preview"

        # Build parts once — images uploaded here, URL refs shared across requests.
        base_parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images is not None:
            if get_number_of_images(images) > 14:
                raise ValueError("The current maximum number of supported images is 14.")
            base_parts.extend(await create_image_parts(cls, images))
        if files is not None:
            base_parts.extend(files)

        image_config = GeminiImageConfig(imageSize=resolution)
        if aspect_ratio != "auto":
            image_config.aspectRatio = aspect_ratio

        gen_config = GeminiImageGenerationConfig(
            responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
            imageConfig=image_config,
            thinkingConfig=GeminiThinkingConfig(thinkingLevel=thinking_level),
        )

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(
                parts=[GeminiTextPart(text=system_prompt)], role=None
            )

        responses = await asyncio.gather(*[
            _single_call(cls, model, base_parts, gen_config, gemini_system_prompt)
            for _ in range(batch_size)
        ])

        image_tensors = []
        texts = []
        for response in responses:
            image_tensors.append(await _get_image_from_response(response))
            texts.append(get_text_from_response(response))

        return IO.NodeOutput(torch.cat(_resize_to_match(image_tensors), dim=0), "\n---\n".join(texts))


NODE_CLASS_MAPPINGS = {
    "NanoBananaBatch": NanoBananaBatch,
    "NanoBanana2Batch": NanoBanana2Batch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaBatch": "Nano Banana Batch",
    "NanoBanana2Batch": "Nano Banana 2 Batch",
}
