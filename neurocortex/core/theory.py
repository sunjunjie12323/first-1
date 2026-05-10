"""
NeuroCortex: Theoretical Framework

This module defines the formal mathematical framework for the four core
innovations of the NeuroCortex system, with explicit differentiation from
existing work.

=== INNOVATION 1: Fragment-Level Reconstructive Recall with Variable Detail ===

Formal Definition:
  Given query q, episodic traces E = {e_1, ..., e_n}, semantic schemas S:
  
  Step 1 - Cue Activation:
    A(q) = {e_i | sim(emb(q), emb(e_i)) * strength(e_i) > theta_cue}
  
  Step 2 - CA3 Spreading Activation:
    A'(q) = A(q) U {e_j | exists e_i in A(q), assoc(e_i, e_j) > phi}
  
  Step 3 - Fragment Detail Determination (NOVEL):
    For each e_i in A'(q), compute combined activation:
      alpha_i = activation(e_i) * (1 + importance(e_i)) * (1 + |emotion(e_i)|)
    
    Detail level:
      detail(e_i) = FULL      if alpha_i > alpha_full
      detail(e_i) = GIST      if alpha_i > alpha_gist
      detail(e_i) = KEYWORD   otherwise
    
    Fragment content:
      frag(e_i) = content(e_i)          if FULL
      frag(e_i) = gist(e_i)             if GIST
      frag(e_i) = keywords(e_i, k=10)   if KEYWORD
  
  Step 4 - LLM Reconstruction (NOVEL):
    R(q) = LLM(q, {frag(e_i) for e_i in A'(q)}, {schema_j for schema_j in S'(q)})
  
  This is fundamentally different from:
    - RAG: R_RAG(q) = {e_i | sim(emb(q), emb(e_i)) > theta} (retrieval only)
    - True Memory (Adler & Zehavi, 2026): Verbatim preservation + retrieval pipeline
    - E-mem (Wang et al., 2026): Multi-agent context reasoning (not fragment reconstruction)
    - CA3Mem (Zhang et al., 2026): Trajectory recombination (not memory fragment reconstruction)

  Key novelty: The VARIABLE DETAIL LEVEL mechanism (Step 3) has no precedent.
  It models the vividness gradient in human recall: strongly activated memories
  are recalled in full detail, while weakly activated ones produce only vague
  impressions (keywords). This directly implements Bartlett's (1932) theory that
  recall is reconstructive, not reproductive.

=== INNOVATION 2: Four-Transmitter Neuromodulatory State Machine ===

Formal Definition:
  State vector: M(t) = [ACh(t), DA(t), 5-HT(t), NE(t)]
  
  Update rules:
    ACh(t+1) = ACh(t) * lambda_decay + novelty(x_t, E) * (1 - lambda_decay)
    DA(t+1)  = DA(t) * lambda_decay + reward(feedback_t) * (1 - lambda_decay)
    5-HT(t+1) = 5-HT(t) * lambda_decay + social(x_t) * (1 - lambda_decay)
    NE(t+1)  = NE(t) * lambda_decay + arousal(novelty, emotion) * (1 - lambda_decay)
  
  Gating functions:
    G_encode(t) = 0.3 + 0.7 * ACh(t)        [encoding gate]
    G_consolidate(t) = 0.2 + 0.8 * DA(t)     [consolidation gate]
    G_social(t) = 0.5 + 0.5 * 5-HT(t)        [social memory gate]
    G_precision(t) = 0.5 + 0.5 * NE(t)       [encoding precision gate]
  
  Differentiation from existing work:
    - ZenBrain (NeurIPS'25): Emotional valence tagging only (single dimension)
    - True Memory (2026): 3-signal encoding gate (novelty, salience, prediction error)
      but no dynamic state machine, no consolidation gating, no social modulation
    - Pirazzini & Ursino (2025): Computational model of ACh in hippocampus
      but not applied to LLM agents, no DA/5-HT/NE integration
  
  Key novelty: The 4-transmitter DYNAMIC STATE MACHINE with JOINT gating of
  encoding, consolidation, social memory, and precision is unprecedented in
  LLM agent memory systems. The ACh-DA complementarity (ACh gates WHAT to
  encode, DA gates WHAT to consolidate) mirrors recent neuroscience findings
  (Zhang et al., 2025, Nature Neuroscience).

=== INNOVATION 3: Dentate Gyrus Adaptive Pattern Separation ===

Formal Definition:
  Given input embedding x and existing embedding matrix W:
    
    max_sim(x, W) = max_i |cos_sim(x, w_i)|
    
    Pattern separation:
      x' = x + epsilon * max_sim(x, W) * N(0, I)
      x' = x' / ||x'|| * ||x||
  
  where epsilon is the separation strength parameter.
  
  Properties:
    1. When max_sim is low (novel input): epsilon * max_sim is small,
       x' approx x (minimal distortion of novel inputs)
    2. When max_sim is high (similar to existing): epsilon * max_sim is large,
       x' is pushed away from existing traces (pattern separation)
    3. The separation strength is ADAPTIVE: it scales with similarity
  
  Differentiation from existing work:
    - HeLa-Mem (Zhu et al., 2026): Hebbian learning (strengthens connections)
      Pattern separation is the OPPOSITE: it makes similar inputs MORE distinct
    - CA3Mem (Zhang et al., 2026): CA3 autoassociative recall
      DG pattern separation is the COMPLEMENT: it happens BEFORE CA3 recall
    - Standard vector DB: No pattern separation at all
  
  Key novelty: The ADAPTIVE noise injection mechanism where separation strength
  scales with maximum similarity to existing traces is a computationally efficient
  approximation of DG pattern separation that has no precedent in LLM memory systems.

=== INNOVATION 4: Memory Distortion Quantification ===

Formal Definition:
  Given a reconstructed memory R(q) with source traces T = {t_1, ..., t_k}
  and source schemas S = {s_1, ..., s_m}:
  
  Distortion score:
    D(R(q)) = w_1 * (1 - mean_activation(T)) * sensitivity
            + w_2 * |spread_traces| * sensitivity
            + w_3 * (|S| - 1) * sensitivity
            + w_4 * (1 - mean_consolidation(T)) * sensitivity
  
  where:
    - mean_activation: average activation of source traces (lower = more gap-filling)
    - spread_traces: number of indirectly activated traces (more = more integration)
    - |S|: number of merged schemas (more = more generalization)
    - mean_consolidation: average consolidation level (lower = more episodic, less stable)
    - sensitivity: distortion sensitivity parameter
  
  Properties:
    D = 0: Perfect reproductive recall (no distortion)
    D = 1: Maximum distortion (completely reconstructed from weak cues)
  
  Differentiation from existing work:
    - NO existing LLM memory system quantifies memory distortion
    - Schacter (2001) classified 7 types of human memory distortion, but
      no computational system has operationalized these as measurable metrics
    - This metric enables empirical comparison between artificial and human
      memory distortion patterns
  
  Key novelty: Memory distortion as a FIRST-CLASS MEASURABLE PROPERTY of an
  artificial memory system is entirely novel. This opens the door to empirical
  validation against human memory distortion patterns.
"""
