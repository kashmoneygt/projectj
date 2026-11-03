import sys

import docker
from swebench.cli import cli
from swebench.image_builder import docker_build

_build_instance_images = docker_build.build_instance_images


def _build_missing(
    client: docker.DockerClient,
    image_specs: list,
    force_rebuild: bool = False,  # ignored (swebench passes True)
    max_workers: int = 4,
    dry_run: bool = False,
) -> tuple[list, list]:
    return _build_instance_images(
        client=client, image_specs=image_specs, force_rebuild=False, max_workers=max_workers, dry_run=dry_run
    )


# swebench rebuilds every image on each eval with --task-repo; ours never changes, so build each once
docker_build.build_instance_images = _build_missing

if __name__ == "__main__":
    sys.exit(cli.main())
