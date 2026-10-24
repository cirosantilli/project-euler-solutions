import ProjectEulerSolutions.Lib.TotientSievePrepareCalls

/-! Source outer iterations preserve the complete linear-sieve table invariant. -/
namespace ProjectEulerSolutions.TotientSieveCalls
open FormalSW.PyAst HeapCalls ProjectEulerStatements.TotientLinearSieve
set_option Elab.async false
variable {program : Module}

theorem outer_continuation (next : Stmt) : Calls.loopContinuation outerBody next = next := rfl

/-- Prime initialization and the complete inner pass advance one source iteration. -/
theorem iteration_result (limit i : Nat) (cache : List Nat) (large : 2 ≤ i)
    (within : i ≤ limit) (ordered : cache.Pairwise (· < ·))
    (complete : ∀ p, p ∈ cache ↔ Nat.Prime p ∧ p < i)
    (base : Mutable.Heap) (env : HeapCalls.Env) (ready : Ready limit i base.length env) :
    ∃ finalEnv, BlockContinues program env outerBody
      (realize base (limit + 1) (expected i) cache) finalEnv
      (realize base (limit + 1) (expected (i + 1)) (nextCache i cache)) ∧
      Ready limit i base.length finalEnv := by
  have advancedComplete := nextCache_complete i cache complete
  have initialized := prepare_result (program := program)
    limit i (expected i) cache large within rfl base env ready
  have primes : ∀ p ∈ nextCache i cache, Nat.Prime p :=
    fun p member => ((advancedComplete p).mp member).1
  obtain ⟨finalEnv, passed, finalReady⟩ := pass_result (program := program)
    limit i (nextCache i cache)
    (prepare i (expected i)) primes within base env ready
  have correct := step_correct limit i (nextCache i cache) (expected i) large within
    (nextCache_ordered i cache ordered complete) advancedComplete (fun _ _ => rfl)
  have same := realize_congr base (limit + 1)
    (pass limit i (nextCache i cache) (prepare i (expected i))) (expected (i + 1))
    (nextCache i cache) (fun n bounded => correct n (by omega))
  rw [same] at passed
  refine ⟨finalEnv, ?_, finalReady⟩
  rw [outer_source]
  exact initialized.append passed

structure Stable (limit address : Nat) (env : HeapCalls.Env) : Prop where
  getLimit : env "limit" = some (number limit)
  getPhi : env "phi" = some (.ref (address + 1))
  getMarks : env "is_comp" = some (.ref (address + 3))
  getCache : env "primes" = some (.ref (address + 4))

theorem Ready.stable {limit i address : Nat} {env : HeapCalls.Env}
    (ready : Ready limit i address env) : Stable limit address env :=
  ⟨ready.getLimit, ready.getPhi, ready.getMarks, ready.getCache⟩

theorem Stable.at_index {limit address : Nat} {env : HeapCalls.Env}
    (stable : Stable limit address env) (i : Nat) :
    Ready limit i address (setLocal env "i" (number i)) where
  getLimit := by simpa [setLocal] using stable.getLimit
  getI := by simp [setLocal]
  getPhi := by simpa [setLocal] using stable.getPhi
  getMarks := by simpa [setLocal] using stable.getMarks
  getCache := by simpa [setLocal] using stable.getCache

/-- The finite source range reaches the complete table for every inclusive bound. -/
theorem outer_loop_result (limit cursor : Nat) (cache : List Nat) (large : 2 ≤ cursor)
    (within : cursor ≤ limit + 1) (ordered : cache.Pairwise (· < ·))
    (complete : ∀ p, p ∈ cache ↔ Nat.Prime p ∧ p < cursor)
    (base : Mutable.Heap) (env : HeapCalls.Env) (stable : Stable limit base.length env) :
    ∃ finalEnv finalCache, BlockContinues program env
      [.rangeStepLoop "i" cursor (limit + 1) 1 outerBody []]
      (realize base (limit + 1) (expected cursor) cache) finalEnv
      (realize base (limit + 1) (expected (limit + 1)) finalCache) ∧
      Stable limit base.length finalEnv := by
  by_cases active : cursor < limit + 1
  · let indexed := setLocal env "i" (number cursor)
    have ready : Ready limit cursor base.length indexed := stable.at_index cursor
    obtain ⟨advancedEnv, advanced, advancedReady⟩ := iteration_result (program := program)
      limit cursor cache large
      (by omega) ordered complete base indexed ready
    have nextComplete : ∀ p, p ∈ nextCache cursor cache ↔ Nat.Prime p ∧ p < cursor + 1 := by
      simpa only [Nat.lt_succ_iff] using nextCache_complete cursor cache complete
    obtain ⟨finalEnv, finalCache, continued, finalStable⟩ :=
      outer_loop_result limit (cursor + 1) (nextCache cursor cache) (by omega) (by omega)
        (nextCache_ordered cursor cache ordered complete) nextComplete base advancedEnv
        advancedReady.stable
    refine ⟨finalEnv, finalCache, ?_, finalStable⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    rw [outer_continuation]
    exact advanced.append continued
  · have finished : cursor = limit + 1 := by omega
    subst cursor
    refine ⟨env, cache, ?_, stable⟩
    apply BlockContinues.range_done (by simp [Calls.rangeContinues])
    exact .nil
termination_by limit + 1 - cursor
decreasing_by omega

end ProjectEulerSolutions.TotientSieveCalls
