"""Reusable slot modules ported from Unified Harness v1.1.0 (harnessstudio.runtime).

These plug into method configs through ModuleRef; none is a registered built-in method yet:
  - selectors.TrajectoryMedoid   (Selector)  <- runtime.modules.MedoidSelector
  - executors.PublicStasis       (Executor)  <- runtime.modules.ProgressMonitor (advisory by default)
  - episode.InstructionRewrite   (EpisodeOp) <- runtime.operators.NRPInstructionOperator + runtime.memory.FrozenMemory
"""
