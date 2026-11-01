import solvers.«61.py.lean.d».Loops

set_option Elab.async false

namespace solvers.«61».dfs
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

private theorem empty_result (fuel : Nat) (values : List Int) (nonempty : values ≠ [])
    (heap : Mutable.Heap) (path remainingAddress prefixCell : Nat)
    (pathFound : heap[path]? = some (.list (numbers values))) (remainingFound : heap[remainingAddress]? = some (.list []))
    (globals : Mutable.Globals) :
    ∃ extra, ReturnsValue program "__formalsw_local_solve_dfs" [.ref path, .ref remainingAddress, .ref prefixCell]
      heap (number (PolygonalCycleSearch.prefixScan (PolygonalCycleSearch.indexed PolygonalFamilies.family) fuel values [])) (heap ++ extra) globals := by
  have model : PolygonalCycleSearch.prefixScan (PolygonalCycleSearch.indexed PolygonalFamilies.family) fuel values [] =
      (if PolygonalCycleSearch.closed values then values.sum else 0) := by cases fuel <;> rfl
  refine ⟨[], ?_⟩
  rw [model]
  simpa only [List.append_nil] using closed_result values nonempty heap path remainingAddress pathFound remainingFound (.ref prefixCell) globals

/-- The full recursive indexed search terminates and returns its exact maximum-search model. -/
theorem search_result (fuel : Nat) (heap : Mutable.Heap) (path remainingAddress prefixCell : Nat) (maxSides : Int)
    (values : List Int) (remaining : List Int)
    (state : SearchState heap path remainingAddress prefixCell maxSides values remaining)
    (nonempty : values ≠ []) (depth : remaining.length ≤ fuel) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "__formalsw_local_solve_dfs" [.ref path, .ref remainingAddress, .ref prefixCell]
      heap (number (PolygonalCycleSearch.prefixScan (PolygonalCycleSearch.indexed PolygonalFamilies.family) fuel values remaining)) (heap ++ extra) globals := by
  induction fuel generalizing heap path remainingAddress prefixCell maxSides values remaining with
  | zero =>
      have empty : remaining = [] := List.eq_nil_of_length_eq_zero (by omega)
      subst remaining
      exact empty_result 0 values nonempty heap path remainingAddress prefixCell state.pathFound state.remainingFound globals
  | succ fuel ih =>
      by_cases empty : remaining = []
      · subst remaining
        exact empty_result (fuel + 1) values nonempty heap path remainingAddress prefixCell state.pathFound state.remainingFound globals
      · obtain ⟨last, atEnd⟩ : ∃ last, values.getLast? = some last := by
          cases found : values.getLast? with
          | none => exact False.elim (nonempty (List.getLast?_eq_none_iff.mp found))
          | some last => exact ⟨last, rfl⟩
        have noTypes : remaining.isEmpty = false := by
          cases remaining with
          | nil => exact False.elim (empty rfl)
          | cons _ _ => rfl
        let params := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
        let sourceBody := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
        let env := functionEnv params sourceBody [.ref path, .ref remainingAddress, .ref prefixCell] globals
        have bound : Bindings env path remainingAddress prefixCell := Bindings.function path remainingAddress prefixCell globals
        let bestEnv := setLocal env "best" (number 0)
        have bestBound : Bindings bestEnv path remainingAddress prefixCell := bound.local "best" _ (by decide)
        let suffixEnv := setLocal bestEnv "suffix" (number (last % 100))
        have suffixBound : Bindings suffixEnv path remainingAddress prefixCell := bestBound.local "suffix" _ (by decide)
        let visit := fun kind value => PolygonalCycleSearch.prefixScan (PolygonalCycleSearch.indexed PolygonalFamilies.family) fuel (values ++ [value])
          (remaining.filter (fun other => decide (other ≠ kind)))
        have child : ∀ currentHeap kind unused value,
            SearchState currentHeap path remainingAddress prefixCell maxSides values remaining → kind ∈ remaining →
            currentHeap[unused]? = some (.list (kinds (remaining.filter (fun other => decide (other ≠ kind))))) →
            value ∈ PolygonalCycleSearch.indexed PolygonalFamilies.family kind (last % 100) → value ∉ values →
            ∃ extra, ReturnsValue program "__formalsw_local_solve_dfs"
              [.ref (currentHeap.length + 1), .ref unused, .ref prefixCell] (childHeap currentHeap values value)
              (number (visit kind value)) (childHeap currentHeap values value ++ extra) globals := by
          intro currentHeap kind unused value ready member unusedFound _ _
          have smaller := PolygonalCycleSearch.unused_shorter remaining kind member
          exact ih (childHeap currentHeap values value) (currentHeap.length + 1) unused prefixCell maxSides
            (values ++ [value]) (remaining.filter (fun other => decide (other ≠ kind)))
            (ready.child kind value unused unusedFound) (by simp) (by omega)
        obtain ⟨extra, finalEnv, executed, finalBound, finalBest, frame⟩ :=
          type_loop state suffixBound (last % 100) rfl 0 (by simp [suffixEnv, bestEnv, setLocal]) visit child
        have model : PolygonalCycleSearch.prefixScan (PolygonalCycleSearch.indexed PolygonalFamilies.family) (fuel + 1) values remaining =
            remaining.foldl (advanceKind values (last % 100) visit) 0 := by
          rw [PolygonalCycleSearch.prefixScan, noTypes]
          simp only [Bool.false_eq_true, if_false, PolygonalCycleSearch.last, atEnd, Option.getD_some]
          rfl
        refine ⟨extra, ?_⟩
        rw [model]
        apply BlockReturns.function rfl rfl
        change BlockReturns program env code heap _ (heap ++ extra) globals
        have tested : Evaluates program env exhausted heap (flag false) heap globals := by
          apply Evaluates.unary (middle := heap) (finalHeap := heap) (Evaluates.lookup bound.getRemaining)
          simp [HeapCalls.unary, Mutable.truth, state.remainingFound, kinds, noTypes, flag]
        apply BlockReturns.branch (flag := false) tested rfl
        apply BlockReturns.assign Evaluates.constant
        apply BlockReturns.assign (suffix_evaluate values last atEnd heap path state.pathFound bestEnv bestBound.getPath globals)
        exact BlockContinues.returns executed (BlockReturns.return_value (Evaluates.lookup finalBest))
end solvers.«61».dfs
