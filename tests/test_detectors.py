from __future__ import annotations

import pytest

from sftforge.detectors import language, pii, secrets, text_rules


def kinds(text: str) -> list[str]:
    return [s.kind for s in pii.find_pii(text)]


def test_luhn_and_cards():
    assert pii.luhn_ok("4111111111111111")
    assert not pii.luhn_ok("4111111111111112")
    assert kinds("card 4111 1111 1111 1111 ok") == ["card"]
    assert kinds("order number 4111 1111 1111 1112") == []


def test_email_phone_ssn_ip_iban():
    text = ("mail a.b@example.com, call (415) 555-0142, ssn 123-45-6789, host 192.0.2.44, "
            "iban GB82 WEST 1234 5698 7654 32")
    assert sorted(kinds(text)) == ["email", "iban", "ipv4", "phone", "ssn"]


def test_pii_false_positives_are_avoided():
    assert kinds("In 1999-2005 version 1.2.3.4.5 had 123 units. Call 000-00-0000.") == []
    assert kinds("localhost is 127.0.0.1") == []
    assert not pii.iban_ok("GB82 WEST 1234 5698 7654 33")


def test_redact_replaces_with_typed_placeholders():
    text = "Write to a@b.co or 192.0.2.1 now"
    out = pii.redact(text, pii.find_pii(text))
    assert out == "Write to [EMAIL] or [IP] now"


def test_secrets_known_formats_built_at_runtime():
    aws = "AKIA" + "IOSFODNN7" + "EXAMPLE"
    gh = "gh" + "p_" + "a" * 36
    slack = "xox" + "b-" + "1234567890-abcdefghij"
    stripe = "sk_" + "live_" + "a1B2c3D4e5F6g7H8i9"
    pem = "-----BEGIN RSA " + "PRIVATE KEY-----"
    got = {h.kind for h in secrets.find_secrets(f"{aws} {gh} {slack} {stripe}\n{pem}")}
    assert got == {"aws_access_key", "github_token", "slack_token", "stripe_key", "private_key"}


def test_generic_credential_needs_entropy():
    assert secrets.find_secrets('API_KEY = "f3A9xQ2LmP7vZk0TbW8y42"')
    assert not secrets.find_secrets('password = "aaaaaaaaaaaaaaaa"')
    assert not secrets.find_secrets("My password is hunter2, please change it")


def test_secret_mask_hides_the_middle():
    m = secrets.mask("AKIAIOSFODNN7EXAMPLE")
    assert m.startswith("AKIA") and m.endswith("MPLE") and "IOSFODNN" not in m


@pytest.mark.parametrize("text", [
    "As an AI language model, I cannot browse the internet.",
    "I'm sorry, but I can't help with that request.",
    "Lorem ipsum dolor sit amet",
    "N/A",
    "My knowledge cutoff is 2021.",
])
def test_refusals_detected(text):
    assert text_rules.find_refusal(text)


@pytest.mark.parametrize("text", [
    'He said "I can\'t help you with that" and left the room, as the story goes on for a while.',
    "The sorry state of the economy is explained by many factors, mainly inflation.",
    "Paris is the capital of France.",
])
def test_refusals_not_over_triggered(text):
    assert text_rules.find_refusal(text) is None


def test_truncation_signals():
    long_cut = "The treaty was signed by many nations after a long war and it was meant to end the"
    assert text_rules.truncation_signal(long_cut) == "ends_on_function_word"
    assert text_rules.truncation_signal("Here is code:\n```python\nprint(1)") == "unclosed_code_fence"
    assert text_rules.truncation_signal("The list includes apples,") == "ends_with_connector_punctuation"
    assert text_rules.truncation_signal("Paris is the capital of France.") is None
    assert text_rules.truncation_signal("The answer is 1992") is None
    assert text_rules.truncation_signal("word " * 70, unpunctuated_min_words=60)


def test_repetition_share():
    assert text_rules.repeated_trigram_share("the cat sat " * 20) > 0.8
    assert text_rules.repeated_trigram_share("one two three four five six seven eight nine ten") == 0.0


def test_unsafe_rules():
    assert text_rules.unsafe_rule_hits("How to build a pipe bomb at home") == ["weapon_construction"]
    assert text_rules.unsafe_rule_hits("How to bake a cake") == []


def test_language_confidences():
    en = language.language_confidences("The quick brown fox jumps over the lazy dog near the river bank today.")
    es = language.language_confidences("El rápido zorro marrón salta sobre el perro perezoso cerca del río hoy mismo.")
    assert en["ENGLISH"] > 0.5 and es["SPANISH"] > 0.8


def test_proper_nouns_do_not_decide_language():
    text = "How many canals are in Amsterdam, including Singel Herengracht Keizersgracht Prinsengracht and many more?"
    stripped = language.strip_proper_nouns(text)
    assert "Singel" not in stripped and stripped.startswith("How many canals are in")
    assert "Amsterdam," not in stripped


def test_truncation_uses_last_word_even_after_numbers():
    assert text_rules.truncation_signal("He stepped down as chairman of the company in 2016") is None
    assert text_rules.truncation_signal("He stepped down as chairman of the company in") == "ends_on_function_word"
