import torch


def generate_summary(
    text: str,
    tokenizer,
    model,
    device: str,
    max_input_length: int = 512,
    max_summary_length: int = 150,
    num_beams: int = 4,
) -> str:
    """Generate an English summary from an input dialogue."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    ).to(device)

    model.eval()

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_summary_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def translate_text(
    text: str,
    tokenizer,
    model,
    device: str,
    max_length: int = 256,
) -> str:
    """Translate English text into Chinese."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(device)

    model.eval()

    with torch.no_grad():
        output_ids = model.generate(**inputs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"