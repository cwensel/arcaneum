"""Deploy planning: desired versus current state for every lifecycle case (RDR-025)."""

import pytest

from arcaneum.embeddings.remote.deploy import ContainerState, Desired, plan_host
from arcaneum.embeddings.remote.pool import EndpointStatus

NEW = "a" * 64
OLD = "b" * 64
SPEC = "spec-1"


def desired(fingerprint=NEW, spec=SPEC):
    return Desired(fingerprint=fingerprint, version="0.11.0", spec_hash=spec)


def container(fingerprint=NEW, spec=SPEC, running=True):
    return ContainerState(running=running, fingerprint=fingerprint, spec_hash=spec, image="x")


def up(fingerprint=NEW):
    return EndpointStatus("dev-0:7300", "up", code_fingerprint=fingerprint)


DOWN = EndpointStatus("dev-0:7300", "unreachable", "unreachable (refused)")


@pytest.mark.parametrize(
    ("case", "current", "images", "status", "want", "refresh", "build", "action"),
    [
        ("first install", None, set(), DOWN, NEW, False, True, "create"),
        ("upgrade", container(OLD), {OLD}, up(OLD), NEW, False, True, "replace"),
        # The workstation went back to OLD, whose image deploy retained.
        ("downgrade", container(NEW), {NEW, OLD}, up(NEW), OLD, False, False, "replace"),
        ("update (development build)", container(OLD), {OLD}, up(OLD), NEW, False, True, "replace"),
        ("reconfigure", container(spec="spec-0"), {NEW}, up(), NEW, False, False, "replace"),
        ("refresh", container(), {NEW}, up(), NEW, True, True, "replace"),
        ("stopped", container(running=False), {NEW}, DOWN, NEW, False, False, "start"),
        ("missing", None, {NEW}, DOWN, NEW, False, False, "create"),
        ("unresponsive", container(), {NEW}, DOWN, NEW, False, False, "replace"),
        ("nothing changed", container(), {NEW}, up(), NEW, False, False, "none"),
    ],
)
def test_lifecycle_table(case, current, images, status, want, refresh, build, action):
    plan = plan_host(current, images, status, desired(want), refresh=refresh)

    assert (plan.build, plan.container) == (build, action), case
    assert plan.verify is (action in ("create", "replace")), case


def test_unresponsive_tailscale_worker_resolving_outside_the_tailnet_is_reported():
    plan = plan_host(
        container(),
        {NEW},
        DOWN,
        desired(),
        refresh=False,
        resolution_problem="dev-0 resolves to 192.168.1.5, outside the tailnet",
    )

    assert plan.container == "none"
    assert not plan.build
    assert "outside the tailnet" in plan.report


def test_running_container_reporting_another_fingerprint_is_replaced():
    plan = plan_host(container(), {NEW}, up(OLD), desired(), refresh=False)

    assert plan.container == "replace"
