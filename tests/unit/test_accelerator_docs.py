import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
USER_GUIDES = (
    ROOT / "README.md",
    ROOT / "commands/index.md",
    ROOT / "docs/guides/accelerators.md",
    ROOT / "docs/guides/cli-reference.md",
    ROOT / "docs/guides/pdf-indexing.md",
    ROOT / "docs/guides/quickstart.md",
)


def test_user_guides_do_not_recommend_unbounded_mps_allocator():
    unsafe_assignment = re.compile(
        r"(?:export\s+)?PYTORCH_MPS_HIGH_WATERMARK_RATIO\s*=\s*0(?:\.0+)?(?![.\d])"
    )
    for path in USER_GUIDES:
        text = path.read_text(encoding="utf-8")
        matches = unsafe_assignment.findall(text)
        # Mentioning the exact value in a prohibition is intentional; executable
        # shell assignments and positive recommendations are not.
        assert not re.search(
            r"export\s+PYTORCH_MPS_HIGH_WATERMARK_RATIO\s*=\s*0(?:\.0+)?(?![.\d])\s*$",
            text,
            re.MULTILINE,
        )
        assert len(matches) <= 1
        if matches:
            assert "never set" in text.lower() or "refuses" in text.lower()


def test_user_guides_do_not_make_blanket_gpu_speed_claims():
    forbidden = (
        "1.5-3x speedup",
        "verified with gpu support",
        "gpu acceleration: opt-in with --gpu for faster",
        "gpu speedup:",
    )
    for path in USER_GUIDES:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(claim in text for claim in forbidden), path


def test_accelerator_guide_links_to_versioned_policy_and_evidence():
    guide = (ROOT / "docs/guides/accelerators.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", guide)
    for target in links:
        assert not target.startswith(("http://", "https://"))
        assert ((ROOT / "docs/guides") / target).resolve().is_file(), target
