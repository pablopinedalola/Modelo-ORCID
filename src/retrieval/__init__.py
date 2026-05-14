"""Módulo de recuperación de candidatos desde APIs externas."""

def get_orcid_client():
    from .orcid_client import ORCIDClient
    return ORCIDClient

def get_openalex_client():
    from .openalex_client import OpenAlexClient
    return OpenAlexClient

def get_ror_client():
    from .ror_client import RORClient
    return RORClient

def get_candidate_ranker():
    from .candidate_ranker import CandidateRanker
    return CandidateRanker

# For backwards compatibility if needed, but better to use the getters
# from .orcid_client import ORCIDClient
# from .openalex_client import OpenAlexClient
# from .ror_client import RORClient
# from .candidate_ranker import CandidateRanker
