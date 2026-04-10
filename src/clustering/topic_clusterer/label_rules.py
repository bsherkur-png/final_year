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

MIGRATION_RELEVANCE_KEYWORDS = [
    "asylum",
    "border",
    "channel",
    "deport",
    "detention",
    "home office",
    "immigration",
    "migrant",
    "migration",
    "net migration",
    "refugee",
    "rwanda",
    "smuggling",
    "small boat",
    "trafficking",
    "visa",
]

TARGET_CLUSTER_CUES = {
    "visa routes": [
        "visa",
        "student visa",
        "work visa",
        "family visa",
        "salary threshold",
        "income threshold",
        "skilled worker",
        "health and care visa",
        "dependants",
        "sponsorship",
    ],
    "boat crossings": [
        "small boat",
        "small boats",
        "channel",
        "channel crossing",
        "boat crossing",
        "crossing",
        "border force",
    ],
    "migration figures": [
        "net migration",
        "figures",
        "stats",
        "official figures",
        "record high",
        "record low",
    ],
    "rwanda plan": [
        "rwanda",
        "rwanda bill",
        "rwanda plan",
        "rwanda scheme",
        "safety of rwanda",
        "flights to rwanda",
    ],
}

EXCLUDED_TITLE_CUES = [
    "australia",
    "biden",
    "california",
    "canada",
    "canadian",
    "congo",
    "congress",
    "israel",
    "meghan",
    "mexico",
    "prince harry",
    "senate",
    "texas",
    "trump",
    "italy",
]


def build_cluster_labels(top_terms_by_cluster: dict[int, list[str]]) -> dict[int, str]:
    labels = {}

    for cluster_id, top_terms in top_terms_by_cluster.items():
        fallback_label = " / ".join(top_terms[:3]) if top_terms else f"Cluster {cluster_id}"
        labels[cluster_id] = MANUAL_TOPIC_LABELS.get(cluster_id, fallback_label)

    return labels


def is_migration_relevant(title: str, processed_title: str = "") -> bool:
    title_candidates = [_normalize_rule_text(title), _normalize_rule_text(processed_title)]

    for keyword in MIGRATION_RELEVANCE_KEYWORDS:
        normalized_keyword = _normalize_rule_text(keyword)
        if normalized_keyword and any(_contains_keyword(candidate, normalized_keyword) for candidate in title_candidates):
            return True

    return False


def is_target_cluster_candidate(title: str, processed_title: str = "") -> bool:
    title_candidates = [_normalize_rule_text(title), _normalize_rule_text(processed_title)]

    for keywords in TARGET_CLUSTER_CUES.values():
        for keyword in keywords:
            normalized_keyword = _normalize_rule_text(keyword)
            if normalized_keyword and any(_contains_keyword(candidate, normalized_keyword) for candidate in title_candidates):
                return True

    return False


def is_title_in_scope(title: str, processed_title: str = "") -> bool:
    title_candidates = [_normalize_rule_text(title), _normalize_rule_text(processed_title)]

    for cue in EXCLUDED_TITLE_CUES:
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
