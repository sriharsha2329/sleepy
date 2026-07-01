"""curvature_wm.env — offline ARC-AGI-3 game play.

Local offline games live in ``environment_files/`` (wired by ``curvature_wm.paths`` via
``ENVIRONMENTS_DIR`` + ``OPERATION_MODE=offline`` — no network/API). ``Live`` drives one game
through the LOCAL perception stack (``curvature_wm/perception/graph_extract``) into our per-object
graphs and node latents (centroids + Mahalanobis footprint):

    from curvature_wm_model.env.live import Live, list_games
"""
