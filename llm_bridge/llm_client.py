import os
from typing import List, Dict, Optional


def _zhipu_api_key() -> str:
    return (
        os.environ.get("ZHIPU_API_KEY", "").strip()
        or os.environ.get("ZHIPUAI_API_KEY", "").strip()
    )


def _wrap_jpeg_data_url(b64: str) -> str:
    if b64.startswith("data:"):
        return b64
    return "data:image/jpeg;base64," + b64


def call_zhipu(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    image_base64: Optional[str] = None,
    images_base64: Optional[List[str]] = None,
) -> str:
    """Zhipu GLM. Set ZHIPU_API_KEY or ZHIPUAI_API_KEY.
    Vision: pass image_base64 (single) or images_base64 (multiple JPEG base64 or data URLs)."""
    try:
        from zhipuai import ZhipuAI
    except ImportError as e:
        raise ImportError("pip install zhipuai") from e

    api_key = _zhipu_api_key()
    if not api_key:
        raise EnvironmentError("Set environment variable ZHIPU_API_KEY (or ZHIPUAI_API_KEY)")

    model = model or os.environ.get("ZHIPU_MODEL", "glm-4-flash").strip()

    client = ZhipuAI(api_key=api_key)

    imgs: List[str] = []
    if images_base64:
        imgs = [_wrap_jpeg_data_url(x) for x in images_base64 if x]
    elif image_base64:
        imgs = [_wrap_jpeg_data_url(image_base64)]

    if imgs:
        content: List[Dict] = [{"type": "text", "text": user}]
        for url in imgs:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
    return resp.choices[0].message.content or ""


def call_dashscope(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    image_url: Optional[str] = None,
) -> str:
    """Aliyun DashScope compatible chat (text or multimodal URL). Set DASHSCOPE_API_KEY."""
    try:
        import dashscope
    except ImportError as e:
        raise ImportError("pip install dashscope") from e

    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("Set environment variable DASHSCOPE_API_KEY")

    dashscope.api_key = key
    model = model or os.environ.get("DASHSCOPE_MODEL", "qwen-turbo").strip()

    from dashscope import MultiModalConversation, Generation

    if image_url:
        messages = [
            {"role": "system", "content": [{"text": system}]},
            {
                "role": "user",
                "content": [{"text": user}, {"image": image_url}],
            },
        ]
        resp = MultiModalConversation.call(model=model, messages=messages)
        if resp.status_code != 200:
            raise RuntimeError("DashScope multimodal error: %s %s" % (resp.code, resp.message))
        content = resp.output.choices[0].message.content
        if isinstance(content, list) and content and isinstance(content[0], dict):
            return content[0].get("text", "") or str(content[0])
        return str(content)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = Generation.call(model=model, messages=messages, result_format="message")
    if resp.status_code != 200:
        raise RuntimeError("DashScope error: %s %s" % (resp.code, resp.message))
    return resp.output.choices[0].message.content


def call_provider(
    provider: str,
    system: str,
    user: str,
    *,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
) -> str:
    p = provider.lower().strip()
    if p in ("zhipu", "glm", "chatglm"):
        return call_zhipu(system, user, image_base64=image_base64 or None)
    if p in ("dashscope", "ali", "aliyun", "qwen"):
        return call_dashscope(system, user, image_url=image_url)
    raise ValueError("Unknown provider: %s (use zhipu or dashscope)" % provider)
