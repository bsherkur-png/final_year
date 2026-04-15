import re


MANUAL_TOPIC_LABELS = {
    0: "Small boats",
    1: "Rwanda Bill ",
    2: "Rwanda Bill",
    3: "Topic 3",
    4: "Visa Bill",
    5: "Topic 5",
    6: "Topic 6",
    7: "Topic 7",
    8: "Topic 8",
    9: "Topic 9",
    10: "Topic 11",
    11: "Topic 12",
    12: "Topic 13",
    13: "Topic 14",
    14: "Topic 15",
}

INCLUSION_KEYWORDS = [
    "rwanda", "asylum", "deport", "deportation", "migrant", "migration",
    "scheme", "plan", "bill", "flights", "home office", "supreme court",
    "appeal", "treaty", "small boats"
]

EXCLUSION_KEYWORDS = [
    "tourism",  "sport", "genocide memorial"
]

ARTICLE_TYPE_EXCLUSIONS = [
    "analysis",
    "explainer",
    "live",
    "opinion",
    "podcast",
    "watch",
]


def build_cluster_labels(top_terms_by_cluster: dict[int, list[str]]) -> dict[int, str]:
    labels = {}

    for cluster_id, top_terms in top_terms_by_cluster.items():
        fallback_label = " / ".join(top_terms[:3]) if top_terms else f"Cluster {cluster_id}"
        labels[cluster_id] = MANUAL_TOPIC_LABELS.get(cluster_id, fallback_label)

    return labels


def is_migration_relevant(title: str, processed_title: str = "") -> bool:
    return is_rwanda_title_relevant(title, processed_title)


def is_rwanda_title_relevant(title: str, processed_title: str = "") -> bool:
    title_candidates = [_normalize_rule_text(title), _normalize_rule_text(processed_title)]
    has_inclusion_match = False
    has_policy_match = False

    for keyword in INCLUSION_KEYWORDS:
        normalized_keyword = _normalize_rule_text(keyword)
        if normalized_keyword and any(_contains_keyword(candidate, normalized_keyword) for candidate in title_candidates):
            has_inclusion_match = True
            if normalized_keyword in ("bill", "scheme", "asylum", "home office", "deport", "treaty"):
                has_policy_match = True

    if not has_inclusion_match:
        return False

    for keyword in EXCLUSION_KEYWORDS:
        normalized_keyword = _normalize_rule_text(keyword)
        if normalized_keyword and any(_contains_keyword(candidate, normalized_keyword) for candidate in title_candidates):
            return False

    for keyword in ARTICLE_TYPE_EXCLUSIONS:
        normalized_keyword = _normalize_rule_text(keyword)
        if normalized_keyword and any(_contains_keyword(candidate, normalized_keyword) for candidate in title_candidates):
            return False

    if any(_contains_keyword(candidate, "kagame") for candidate in title_candidates) and not has_policy_match:
        return False

    return True


def is_target_cluster_candidate(title: str, processed_title: str = "") -> bool:
    return is_rwanda_title_relevant(title, processed_title)


def is_title_in_scope(title: str, processed_title: str = "") -> bool:
    title_candidates = [_normalize_rule_text(title), _normalize_rule_text(processed_title)]

    for cue in EXCLUSION_KEYWORDS:
        normalized_cue = _normalize_rule_text(cue)
        if normalized_cue and any(_contains_keyword(candidate, normalized_cue) for candidate in title_candidates):
            return False

    for cue in ARTICLE_TYPE_EXCLUSIONS:
        normalized_cue = _normalize_rule_text(cue)
        if normalized_cue and any(_contains_keyword(candidate, normalized_cue) for candidate in title_candidates):
            return False

    return True


def _normalize_rule_text(text: str) -> str:
    if text is None:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _contains_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None
