from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig


def load_mbart_tokenizer(
    model_checkpoint: str = "facebook/mbart-large-50-many-to-many-mmt",
    src_lang: str = "en_XX",
    tgt_lang: str = "en_XX",
):
    """Load mBART tokenizer with source and target language settings."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    return tokenizer


def load_mbart_model(
    model_checkpoint: str = "facebook/mbart-large-50-many-to-many-mmt",
    tokenizer=None,
    tgt_lang: str = "en_XX",
):
    """Load mBART model and set forced beginning-of-sentence token."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    if tokenizer is not None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
        model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id[
            tgt_lang
        ]

    return model


def load_mbart_model_and_tokenizer(
    model_checkpoint: str = "facebook/mbart-large-50-many-to-many-mmt",
    src_lang: str = "en_XX",
    tgt_lang: str = "en_XX",
):
    """Load mBART model and tokenizer together."""
    tokenizer = load_mbart_tokenizer(
        model_checkpoint=model_checkpoint,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    model = load_mbart_model(
        model_checkpoint=model_checkpoint,
        tokenizer=tokenizer,
        tgt_lang=tgt_lang,
    )

    return model, tokenizer