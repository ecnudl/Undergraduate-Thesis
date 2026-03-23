"""
All 16 management operation classes.

Groups:
  - Episodic consolidation: ClusterMerge, TrajectoryToWorkflow, CrossTaskGeneralize, ReindexRelations
  - Deduplication: SignatureDedup, SemanticDedup, CrossTypeDedup, ConflictDetection
  - Failure adjustment: PenalizeOnFailure, BoostOnSuccess, ReflectionCorrection, DynamicDiscard
  - Maintenance: AccessStatsUpdate, TimeDecay, ScoreBasedPrune, QualityCuration
"""

# Episodic consolidation
from .cluster_merge import ClusterMergeOp
from .trajectory_to_workflow import TrajectoryToWorkflowOp
from .cross_task_generalize import CrossTaskGeneralizeOp
from .reindex_relations import ReindexRelationsOp

# Deduplication
from .signature_dedup import SignatureDedupOp
from .semantic_dedup import SemanticDedupOp
from .cross_type_dedup import CrossTypeDedupOp
from .conflict_detection import ConflictDetectionOp

# Failure adjustment
from .penalize_on_failure import PenalizeOnFailureOp
from .boost_on_success import BoostOnSuccessOp
from .reflection_correction import ReflectionCorrectionOp
from .dynamic_discard import DynamicDiscardOp

# Maintenance
from .access_stats_update import AccessStatsUpdateOp
from .time_decay import TimeDecayOp
from .score_based_prune import ScoreBasedPruneOp
from .quality_curation import QualityCurationOp

__all__ = [
    "ClusterMergeOp",
    "TrajectoryToWorkflowOp",
    "CrossTaskGeneralizeOp",
    "ReindexRelationsOp",
    "SignatureDedupOp",
    "SemanticDedupOp",
    "CrossTypeDedupOp",
    "ConflictDetectionOp",
    "PenalizeOnFailureOp",
    "BoostOnSuccessOp",
    "ReflectionCorrectionOp",
    "DynamicDiscardOp",
    "AccessStatsUpdateOp",
    "TimeDecayOp",
    "ScoreBasedPruneOp",
    "QualityCurationOp",
]
