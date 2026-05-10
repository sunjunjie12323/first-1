from __future__ import annotations

import asyncio
import logging
import sys

from neurocortex.api.server import create_app
from neurocortex.config import Config

import uvicorn


def main():
    config = Config()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    logger = logging.getLogger("neurocortex")
    logger.info(f"Starting NeuroCortex with config: {config.to_dict()}")

    app = create_app(
        llm_base_url=config.llm_base_url,
        llm_model=config.llm_model,
        llm_api_type=config.llm_api_type,
        data_dir=config.data_dir,
        system_identity=config.system_identity,
        consolidation_interval=config.consolidation_interval,
    )

    uvicorn.run(
        app,
        host=config.server_host,
        port=config.server_port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
