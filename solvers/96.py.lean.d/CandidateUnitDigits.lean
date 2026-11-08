import solvers.«96.py.lean.d».CandidateOnlyPlaceStep

namespace solvers.«96».reduce_cands.onlyPlace
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev digitLoop (k : Nat) : Stmt := .rangeStepLoop "k" k 9 1 places.digitBody []

theorem digit_resume (next : Stmt) : Calls.loopContinuation places.digitBody next = next := rfl
theorem unit_source : places.unitBody = [.forIn (.name "k") (.call (.name "range") [.constant (.int 9)]) places.digitBody []] := rfl

/-- Every digit of a nine-cell unit is inspected, with early failure and live candidate reads. -/
theorem digits (address unit : Nat) (unitCells : Fin 9 → Cell)
    (peers : ∀ i j, i ≠ j → Peer (unitCells i) (unitCells j)) (different : address ≠ unit)
    (k remaining : Nat) (finish : k + remaining = 9) (state : Propagation.State)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap state.candidates) (locals : Locals address state env)
    (unitFound : heap[unit]? = some (.list (peerLoop.encoded ((List.finRange 9).map unitCells))))
    (getUnit : env "unit" = some (.ref unit)) (nonempty : Propagation.Nonempty state) :
    ∃ finalHeap finalEnv result, heap.length ≤ finalHeap.length ∧ finalHeap.length ≤ heap.length + remaining ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (∀ name, name ∉ ["k", "dbit", "places", "changed"] → finalEnv name = env name) ∧
      OnlyPlace.Conclusion state result ∧
      outcome address result env heap [digitLoop k] finalEnv finalHeap globals := by
  induction remaining generalizing k state heap env with
  | zero =>
      have last : k = 9 := by omega
      subst k
      exact ⟨heap, env, some state, Nat.le_refl _, by omega, by intros; rfl, by intros; rfl,
        OnlyPlace.Success.refl state nonempty, found, locals, BlockContinues.range_done rfl .nil⟩
  | succ remaining ih =>
      have inside : k < 9 := by omega
      let digit : Digit := ⟨k, inside⟩
      let ready := setLocal env "k" (number k)
      have readyLocals : Locals address state ready :=
        ⟨by simpa [ready, setLocal] using locals.candidates, by simpa [ready, setLocal] using locals.changed⟩
      obtain ⟨middleHeap, middleEnv, length, frame, envFrame, executed⟩ := execute address unit state
        ((List.finRange 9).map unitCells) digit heap ready globals found readyLocals unitFound
        (by simpa [ready, setLocal] using getUnit) (by simp [ready, setLocal, digit])
      have conclusion := OnlyPlace.conclusion state unitCells digit peers nonempty
      have names : ∀ name, name ∉ ["k", "dbit", "places", "changed"] → middleEnv name = env name := by
        intro name outside
        rw [envFrame name (fun member => outside (List.mem_cons_of_mem _ member))]
        have away : name ≠ "k" := fun same => outside (by simp [same])
        simp [ready, setLocal, away]
      have continues : Calls.rangeContinues (k : Int) 9 1 = true := by simp [Calls.rangeContinues]; omega
      cases result : OnlyPlace.inspect state ((List.finRange 9).map unitCells) digit with
      | none =>
          simp only [result, outcome] at executed
          simp only [result] at conclusion
          refine ⟨middleHeap, middleEnv, none, by omega, by omega, frame, names, conclusion, ?_⟩
          intro following
          apply BlockReturns.range_step continues
          simpa only [List.append_assoc, List.singleton_append] using
            executed (Calls.loopContinuation places.digitBody (.rangeStepLoop "k" ((k : Int) + 1) 9 1 places.digitBody []) :: following)
      | some middle =>
          simp only [result, outcome] at executed
          simp only [result, OnlyPlace.Conclusion] at conclusion
          have unitKept := (frame unit (List.getElem?_eq_some_iff.mp unitFound).1 different).trans unitFound
          have lookup := (names "unit" (by decide)).trans getUnit
          obtain ⟨finalHeap, finalEnv, finalResult, lower, upper, restFrame, restNames, finalConclusion, finished⟩ :=
            ih (k + 1) (by omega) middle middleHeap middleEnv executed.1 executed.2.1 unitKept lookup conclusion.nonempty
          refine ⟨finalHeap, finalEnv, finalResult, by omega, by omega, ?_, ?_, OnlyPlace.Conclusion.trans conclusion finalConclusion, ?_⟩
          · intro index bound away
            exact (restFrame index (by omega) away).trans (frame index bound away)
          · intro name outside
            exact (restNames name outside).trans (names name outside)
          · cases finalResult with
            | none =>
                intro following
                apply BlockReturns.range_step continues
                simpa only [digit_resume, digitLoop, Int.natCast_add, Int.natCast_one, List.append_assoc, List.singleton_append]
                  using executed.2.2.returns (finished following)
            | some after =>
                refine ⟨finished.1, finished.2.1, ?_⟩
                apply BlockContinues.range_step continues
                simpa only [digit_resume, digitLoop, Int.natCast_add, Int.natCast_one]
                  using executed.2.2.append finished.2.2

theorem unit (address location : Nat) (unitCells : Fin 9 → Cell)
    (peers : ∀ i j, i ≠ j → Peer (unitCells i) (unitCells j)) (different : address ≠ location)
    (state : Propagation.State) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap state.candidates) (locals : Locals address state env)
    (unitFound : heap[location]? = some (.list (peerLoop.encoded ((List.finRange 9).map unitCells))))
    (getUnit : env "unit" = some (.ref location)) (nonempty : Propagation.Nonempty state) :
    ∃ finalHeap finalEnv result, heap.length ≤ finalHeap.length ∧ finalHeap.length ≤ heap.length + 9 ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (∀ name, name ∉ ["k", "dbit", "places", "changed"] → finalEnv name = env name) ∧
      OnlyPlace.Conclusion state result ∧
      outcome address result env heap places.unitBody finalEnv finalHeap globals := by
  obtain ⟨finalHeap, finalEnv, result, lower, upper, frame, names, conclusion, executed⟩ :=
    digits address location unitCells peers different 0 9 rfl state heap env globals found locals unitFound getUnit nonempty
  refine ⟨finalHeap, finalEnv, result, lower, upper, frame, names, conclusion, ?_⟩
  cases result with
  | none =>
      intro following
      rw [unit_source]
      exact BlockReturns.for_range (.cons Evaluates.constant .nil) rfl (executed following)
  | some after =>
      refine ⟨executed.1, executed.2.1, ?_⟩
      rw [unit_source]
      exact BlockContinues.for_range (.cons Evaluates.constant .nil) rfl executed.2.2

end
end solvers.«96».reduce_cands.onlyPlace
