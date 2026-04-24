from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_bart_tokenizer(model_checkpoint: str = "facebook/bart-large"):
    """Load BART tokenizer."""
    return AutoTokenizer.from_pretrained(model_checkpoint)


def load_bart_model(model_checkpoint: str = "facebook/bart-large"):
    """Load BART sequence-to-sequence model."""
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


def load_bart_model_and_tokenizer(
    model_checkpoint: str = "facebook/bart-large",
):
    """Load BART model and tokenizer together."""
    tokenizer = load_bart_tokenizer(model_checkpoint)
    model = load_bart_model(model_checkpoint)

    return model, tokenizer