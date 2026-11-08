import solvers.«96.py.lean.d».CandidateUnitTable

namespace solvers.«96».reduce_cands.onlyPlace
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev unitsLoop (table : UnitTable) (index : Nat) : Stmt :=
  .sequenceStepLoop "unit" (.ref table.root) index places.unitBody []
abbrev statement : Stmt := .forIn (.name "unit") (.name "UNITS") places.unitBody []

theorem phase_source : outerBody.drop 3 = [statement] := rfl
theorem unit_resume (next : Stmt) : Calls.loopContinuation places.unitBody next = next := rfl

theorem units (table : UnitTable) (address : Nat) (before remaining : List UnitEntry)
    (split : table.units = before ++ remaining) (state : Propagation.State)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap state.candidates) (locals : Locals address state env)
    (tableView : table.View heap) (separate : table.Separate address) (nonempty : Propagation.Nonempty state) :
    ∃ finalHeap finalEnv result, heap.length ≤ finalHeap.length ∧ finalHeap.length ≤ heap.length + 9 * remaining.length ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (∀ name, name ∉ ["unit", "k", "dbit", "places", "changed"] → finalEnv name = env name) ∧
      OnlyPlace.Conclusion state result ∧
      outcome address result env heap [unitsLoop table before.length] finalEnv finalHeap globals := by
  induction remaining generalizing before state heap env with
  | nil =>
      refine ⟨heap, env, some state, Nat.le_refl _, by simp, by intros; rfl, by intros; rfl,
        OnlyPlace.Success.refl state nonempty, found, locals, ?_⟩
      apply BlockContinues.sequence_done (items := UnitTable.references table.units)
      · simp [Mutable.sequence, sequenceValue, tableView.root]
      · simp [UnitTable.references, split]
      · exact .nil
  | cons current rest ih =>
      have member : current ∈ table.units := by rw [split]; simp
      let ready := setLocal env "unit" (.ref current.address)
      have readyLocals : Locals address state ready :=
        ⟨by simpa [ready, setLocal] using locals.candidates, by simpa [ready, setLocal] using locals.changed⟩
      obtain ⟨middleHeap, middleEnv, middleResult, lower, upper, frame, envFrame, conclusion, executed⟩ :=
        unit address current.address current.cells current.related (separate.units current member) state heap ready globals
          found readyLocals (tableView.units current member) (by simp [ready, setLocal]) nonempty
      have names : ∀ name, name ∉ ["unit", "k", "dbit", "places", "changed"] → middleEnv name = env name := by
        intro name outside
        rw [envFrame name (fun item => outside (List.mem_cons_of_mem _ item))]
        have away : name ≠ "unit" := fun same => outside (by simp [same])
        simp [ready, setLocal, away]
      have sequence : Mutable.sequence (sequenceValue (.ref table.root)) heap = some (.ok (UnitTable.references table.units), heap) := by
        simp [Mutable.sequence, sequenceValue, tableView.root]
      have item : (UnitTable.references table.units)[before.length]? = some (.ref current.address) := by
        simp [UnitTable.references, split]
      cases middleResult with
      | none =>
          change OnlyPlace.Conclusion state none at conclusion
          change outcome address none ready heap places.unitBody middleEnv middleHeap globals at executed
          refine ⟨middleHeap, middleEnv, none, lower, by simp only [List.length_cons]; omega, frame, names, conclusion, ?_⟩
          intro following
          apply BlockReturns.sequence_step sequence item
          simpa only [List.append_assoc, List.singleton_append] using
            executed (Calls.loopContinuation places.unitBody (unitsLoop table (before.length + 1)) :: following)
      | some middle =>
          change OnlyPlace.Success state middle at conclusion
          change outcome address (some middle) ready heap places.unitBody middleEnv middleHeap globals at executed
          obtain ⟨finalHeap, finalEnv, finalResult, restLower, restUpper, restFrame, restNames, finalConclusion, finished⟩ :=
            ih (before ++ [current]) (by simpa only [List.append_assoc, List.singleton_append] using split) middle middleHeap middleEnv
              executed.1 executed.2.1 (table.preserve address heap middleHeap tableView separate frame) conclusion.nonempty
          refine ⟨finalHeap, finalEnv, finalResult, by omega, by simp only [List.length_cons]; omega,
            ?_, ?_, OnlyPlace.Conclusion.trans conclusion finalConclusion, ?_⟩
          · intro index bound away
            exact (restFrame index (by omega) away).trans (frame index bound away)
          · intro name outside
            exact (restNames name outside).trans (names name outside)
          · cases finalResult with
            | none =>
                intro following
                apply BlockReturns.sequence_step sequence item
                simpa only [unit_resume, unitsLoop, List.length_append, List.length_singleton,
                  List.append_assoc, List.singleton_append] using executed.2.2.returns (finished following)
            | some after =>
                refine ⟨finished.1, finished.2.1, ?_⟩
                apply BlockContinues.sequence_step sequence item
                simpa only [unit_resume, unitsLoop, List.length_append, List.length_singleton]
                  using executed.2.2.append finished.2.2

/-- Execute the complete only-choice phase over the represented unit table. -/
theorem phase (table : UnitTable) (address : Nat) (state : Propagation.State)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap state.candidates) (locals : Locals address state env)
    (tableView : table.View heap) (getUnits : env "UNITS" = some (.ref table.root))
    (nonempty : Propagation.Nonempty state) :
    ∃ finalHeap finalEnv result, heap.length ≤ finalHeap.length ∧ finalHeap.length ≤ heap.length + 9 * table.units.length ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (∀ name, name ∉ ["unit", "k", "dbit", "places", "changed"] → finalEnv name = env name) ∧
      OnlyPlace.Conclusion state result ∧
      outcome address result env heap [statement] finalEnv finalHeap globals := by
  obtain ⟨finalHeap, finalEnv, result, lower, upper, frame, names, conclusion, executed⟩ :=
    units table address [] table.units rfl state heap env globals found locals tableView
      (table.separation address heap state.candidates tableView found) nonempty
  refine ⟨finalHeap, finalEnv, result, lower, upper, frame, names, conclusion, ?_⟩
  cases result with
  | none =>
      intro following
      apply BlockReturns.for_sequence (items := UnitTable.references table.units) (by intros; simp) (Evaluates.lookup getUnits)
        (by simp [Mutable.sequence, tableView.root])
      exact executed following
  | some after =>
      refine ⟨executed.1, executed.2.1, ?_⟩
      apply BlockContinues.for_sequence (items := UnitTable.references table.units) (by intros; simp) (Evaluates.lookup getUnits)
        (by simp [Mutable.sequence, tableView.root])
      exact executed.2.2

end
end solvers.«96».reduce_cands.onlyPlace
