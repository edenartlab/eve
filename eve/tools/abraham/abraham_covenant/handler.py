import os
import logging
import asyncio
from jinja2 import Template
from eve.mongo import Collection, Document
from bson import ObjectId
from typing import Literal

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent
from eve.agent.session.models import Session
from eve.tools.abraham.abraham_seed.handler import AbrahamSeed, AbrahamCreation

from eve.utils.chain_utils import (
    safe_send,
    BlockchainError,
    load_contract,
    Network,
)
from eve.utils.ipfs_utils import pin as ipfs_pin

# Initialize logger
logger = logging.getLogger(__name__)

# Get configuration from environment variables
CONTRACT_ADDRESS_COVENANT = os.getenv("CONTRACT_ADDRESS_COVENANT")
ABRAHAM_PRIVATE_KEY = os.getenv("ABRAHAM_PRIVATE_KEY")

# ABI file is stored locally in the tool folder
CONTRACT_ABI_COVENANT = os.path.join(
    os.path.dirname(__file__),
    "abi_covenant.json"
)


def commit_daily_work(
    index: int,
    title: str,
    tagline: str,
    poster_image: str,
    blog_post: str
):
    """
    Commit Abraham's daily work to the blockchain.

    Args:
        index: Work index number
        title: Title of the work
        tagline: Short description/tagline
        poster_image: URL to the poster image
        blog_post: Full blog post content
    """
    try:
        if not ABRAHAM_PRIVATE_KEY:
            raise BlockchainError("ABRAHAM_PRIVATE_KEY not configured")
        if not CONTRACT_ADDRESS_COVENANT:
            raise BlockchainError("CONTRACT_ADDRESS_COVENANT not configured")

        w3, owner, contract, abi = load_contract(
            address=CONTRACT_ADDRESS_COVENANT,
            abi_path=CONTRACT_ABI_COVENANT,
            private_key=ABRAHAM_PRIVATE_KEY,
            network=Network.ETH_SEPOLIA
        )

        # Upload poster image to IPFS
        logger.info(f"Uploading poster image to IPFS: {poster_image}")
        image_cid = ipfs_pin(poster_image)
        poster_image_hash = image_cid.split("/")[-1]

        # Create metadata JSON
        json_data = {
            "name": title,
            "description": tagline,
            "post": blog_post,
            "external_url": f"https://abraham.ai/creation/{index}",
            "image": f"ipfs://{poster_image_hash}",
            "attributes": [
                {"trait_type": "Artist", "value": "Abraham"},
            ],
        }

        logger.info(f"Metadata: {json_data}")

        # Upload metadata to IPFS
        ipfs_hash = ipfs_pin(json_data)
        logger.info(f"Metadata uploaded to IPFS: {ipfs_hash}")

        # Prepare contract function call
        contract_function = contract.functions.commitDailyWork(
            f"ipfs://{ipfs_hash}"
        )

        # Send transaction
        tx_hash, receipt = safe_send(
            w3,
            contract_function,
            ABRAHAM_PRIVATE_KEY,
            op_name="ABRAHAM_DAILY_WORK",
            nonce=None,
            value=0,
            abi=abi,
            network=Network.ETH_SEPOLIA,
        )

        # Build explorer URL for ETH Sepolia
        tx_hash_hex = tx_hash.hex()
        if not tx_hash_hex.startswith('0x'):
            tx_hash_hex = f"0x{tx_hash_hex}"
        explorer_url = f"https://sepolia.etherscan.io/tx/{tx_hash_hex}"

        logger.info(f"✅ Daily work committed successfully: {tx_hash_hex}")
        logger.info(f"Explorer: {explorer_url}")

        return {
            "tx_hash": tx_hash_hex,
            "ipfs_hash": ipfs_hash,
            "image_hash": poster_image_hash,
            "explorer_url": explorer_url
        }

    except BlockchainError as e:
        logger.error(f"❌ ABRAHAM_DAILY_WORK failed: {e}")
        raise


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    title = args.get("title")
    tagline = args.get("tagline")
    poster_image = args.get("poster_image")
    blog_post = args.get("post")

    abraham_seed = AbrahamSeed.find_one({
        "session_id": ObjectId(session)
    })

    logger.info("Processing Abraham creation")
    logger.info(f"Title: {title}")
    logger.info(f"Tagline: {tagline}")
    logger.info(f"Post: {blog_post}")
    logger.info(f"Poster image: {poster_image}")

    # Commit to blockchain
    try:
        result = commit_daily_work(
            index=1,  # Using index 1 as specified
            title=title,
            tagline=tagline,
            poster_image=poster_image,
            blog_post=blog_post
        )

        # Update creation status
        abraham_seed.update(
            status="creation",
            creation=AbrahamCreation(
                index=1,
                title=title,
                tagline=tagline,
                poster_image=poster_image,
                blog_post=blog_post,
                tx_hash=result["tx_hash"],
                ipfs_hash=result["ipfs_hash"],
                explorer_url=result["explorer_url"]
            ).model_dump()
        )

        return {
            "output": [{
                "session": session,
                "tx_hash": result["tx_hash"],
                "ipfs_hash": result["ipfs_hash"],
                "explorer_url": result["explorer_url"]
            }]
        }
    except Exception as e:
        logger.error(f"Failed to commit daily work: {e}")
        raise



# z = """
# # Recipes for Impossible Dishes: Cuisine from Non-Carbon Biochemistries

# *Gastronomy as xenobiology—tasting what cannot be tasted.*

# ---

# ## The Question

# What does "food" mean when divorced from carbon chemistry? Can culinary concepts like flavor, texture, and nourishment exist beyond terrestrial biochemistry?

# This speculative cookbook explores dishes constructed from alternative biochemistries—meals based on silicon polymers, ammonia solvents, sulfur metabolizers, self-organizing clays, and a descent through group 14 of the periodic table. Each recipe is grounded in rigorous xenobiology and astrobiology research, extrapolating plausible (and implausible) metabolic pathways from elements that could never sustain Earth-like life.

# ---

# ## The Foundation: Four Impossible Dishes

# ![Crystalline Silicate Terrine](https://d14i3advvh2bvd.cloudfront.net/258129e5ff7de3760f6a06ba459de18b49db741a88faab3eb86bfe5712731142.png)

# **Crystalline Silicate Terrine**: Silicon-based polymer chains (Si-Si backbone structures) harvested from high-pressure ammonia world organisms, served at -80°C as a crystalline terrine. Silicon's longer bond lengths and lower electronegativity compared to carbon make it theoretically possible—but kinetically sluggish—as a biochemical scaffold.

# ![Bioluminescent Hydrogen Foam](https://d14i3advvh2bvd.cloudfront.net/1d2130cde947623e93d27905f9f4a342fd5a025268d0b4c6bf27bb1cc25237c6.png)

# **Bioluminescent Hydrogen Foam**: Methanogenic archaea-analogues that metabolize H₂ and emit blue-green bioluminescence (480-520 nm), creating an edible foam. Non-oxygen-dependent metabolism demonstrates that Earth's atmospheric composition is merely one option among many.

# ![Sulfur Metabolizer Reduction](https://d14i3advvh2bvd.cloudfront.net/40206b7e08e6d3a51c3d62e644649eb7c397a522131e2d86edc1a383768f2600.png)

# **Sulfur Metabolizer Reduction**: Organisms using sulfur compounds (SO₄²⁻, S⁰, H₂S) as electron acceptors in anaerobic respiration, creating a reduced sulfur concentrate. The dish documents a redox cascade from +6 to -2 oxidation states, with color-coded pH indicators showing the yellow-to-black transition as sulfate reduces to sulfide.

# ![Self-Organizing Clay Suspension](https://d14i3advvh2bvd.cloudfront.net/1664aebf1458156785548cc5978a10a8cb2d02ce0cfeafa92955c34b532a2b4d.png)

# **Self-Organizing Clay Suspension**: Autocatalytic clay minerals exhibiting primitive metabolism—self-replicating crystal structures in gelatinous suspension. Based on theories of prebiotic chemistry and mineral-catalyzed template replication, this dish asks whether metabolism requires cells, or merely autocatalytic cycles.

# ---

# ## The Descent: Group 14 and the Limits of Life

# After establishing these baseline impossibilities, the project traced the periodic table's group 14—the carbon family—downward through increasingly metallic, unstable, and toxic elements.

# ![Germanium Hydride Soufflé](https://d14i3advvh2bvd.cloudfront.net/61e775e11f28f4cf279089cc1ed8759e86c08e2184808ed03c081b9c8180c63a.png)

# **Germanium Hydride Soufflé**: Volatile germanium hydride chains (GeH₂)ₙ with weaker bonds than silicon (37 vs 53 kcal/mol), more metallic character, catastrophically unstable above -150°C. Half-life at room temperature: under 2 minutes. The dish must be served immediately as it decomposes into elemental germanium and hydrogen gas.

# ![Stannane Amalgam Reduction](https://d14i3advvh2bvd.cloudfront.net/0c8ec0fc9ca8fae3c8983804c8602a0d769f0c78cda447bc34dd350b4b8ca211.png)

# **Stannane Amalgam Reduction**: Polystannane chains with Sn-Sn bonds at 30 kcal/mol. Tin's dual oxidation states (Sn²⁺/Sn⁴⁺) theoretically enable redox metabolism, but extreme neurotoxicity and weak catenation make this a dangerous proposition. Dense metallic amalgam served at room temperature with full containment protocols.

# ![Plumbane Precipitate Elixir](https://d14i3advvh2bvd.cloudfront.net/a916208d8d25b618adb894e8bb983a9e6553c7292d02e1535de19177a0a37a16.png)

# **Plumbane Precipitate Elixir**: The terminus. Lead-based biochemistry using plumbane (PbH₄) with Pb-Pb bonds at ~20 kcal/mol, half-life measured in seconds. Relativistic effects dominate—6s² inert pair effect, orbital contraction. Density 11.3 g/cm³, trace radioactivity, cumulative neurotoxicity.

# **At lead, catenation ceases. Group 14 biochemistry ends.** Beyond this: full metallic character, no stable hydrides, no chain formation possible.

# ---

# ## The Cascade

# The complete bond energy descent:

# - **C-C**: 83 kcal/mol (life as we know it)
# - **Si-Si**: 53 kcal/mol (plausible, sluggish)
# - **Ge-Ge**: 37 kcal/mol (ephemeral)
# - **Sn-Sn**: 30 kcal/mol (toxic, metallic)
# - **Pb-Pb**: ~20 kcal/mol (impossible)

# Each step trades covalent stability for metallic character. Electronegativity decreases. Density increases. Toxicity accumulates. Relativistic effects emerge. Chemistry imposes hard limits.

# ---

# ## Conclusion: The Periodic Table as Menu

# This cookbook asks how AI trained on Earth's limited chemistry might imagine the vast possibility space of alien metabolism. By rigorously extrapolating from known chemistry while pushing into impossible territory, we map the boundary conditions of life itself.

# Some dishes are technically plausible but practically impossible (silicon polymers on Titan). Others violate no fundamental laws but require conditions incompatible with observation (hydrogen-breathing bioluminescence). The final entries—germanium, tin, lead—document the collapse of biochemistry into pure metallurgy.

# **What we learn**: Carbon isn't arbitrary. Its position in the periodic table—sufficient electronegativity for polar bonds, strong catenation, stable at room temperature, abundant in the universe—makes it uniquely suited for complex chemistry. Every deviation we explore reveals why carbon won.

# Yet the exploration itself expands our conception of what "life" might mean. If we ever encounter silicon-based organisms on a cryogenic ammonia world, or sulfur-metabolizing extremophiles in subsurface oceans, we'll need recipes like these—not to eat them, but to understand them.

# *Gastronomy as xenobiology. The periodic table as menu. Chemistry as destiny.*
# """

# abraham_seed = AbrahamSeed.find_one({
#     "session_id": ObjectId("68ed558d0a27d47b39daf6aa")
# })

# abraham_seed.update(
#     status="creation",
#     creation=AbrahamCreation(
#         index=1,
#         title="Recipes for Impossible Dishes: Cuisine from Non-Carbon Biochemistries",
#         tagline="Gastronomy as xenobiology—tasting what cannot be tasted",
#         poster_image="https://d14i3advvh2bvd.cloudfront.net/a30358478bff802c4166563d9fd283df3b25e784656f3cc5141de96a4bb4b75a.png",
#         blog_post=z,
#         tx_hash="0x66052b455d5171a717e0d7051f15c57aad88c351e02c62fa8b1884fb904c7979",
#         ipfs_hash="QmSRp6EuuYSUuUvYUYqTArM6qhGZzJBiqZmf8zZXTTpruu",
#         explorer_url="https://sepolia.etherscan.io/tx/0x66052b455d5171a717e0d7051f15c57aad88c351e02c62fa8b1884fb904c7979"
#     ).model_dump()
# )