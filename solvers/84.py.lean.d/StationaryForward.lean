import solvers.«84.py.lean.d».StationaryIteration

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

/-- The actual outer range reaches the final triangular invariant after exactly 119 columns. -/
theorem forward_scan {sides k : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) k) (bound : k ≤ 119)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View state.matrix heap root addresses) (privateRows : ∀ row, root < addresses row)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getSize : env "size" = some (.scalar (.int 120)))
    (getPrevious : env "previous" = some (.scalar (.int state.previous))) :
    ∃ (after : Bareiss.State (Monopoly.augmentedMatrix sides) 119) (afterEnv : HeapCalls.Env) (afterHeap : Mutable.Heap) (afterAddresses : Fin 120 → Nat),
      Bareiss.Runs state after ∧
      BlockContinues program env [.rangeStepLoop "column" k 119 1 eliminationBody []] heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View after.matrix afterHeap root afterAddresses ∧
      (∀ row, root < afterAddresses row) ∧
      afterEnv "previous" = some (.scalar (.int after.previous)) ∧
      (∀ name, name ∉ ["column", "pivot_row", "pivot_values", "pivot", "row", "j", "factor", "previous"] → afterEnv name = env name) ∧
      ∀ address, address < root → afterHeap[address]? = heap[address]? := by
  by_cases active : k < 119
  · let boundEnv := setLocal env "column" (.scalar (.int k))
    obtain ⟨next, nextEnv, nextHeap, nextAddresses, step, ran, nextView, nextPrivate, nextPrevious, nextLocals, nextObjects⟩ :=
      elimination_iteration state (by omega) heap root addresses view privateRows boundEnv globals
        (by simpa [boundEnv, setLocal] using getMatrix) rfl
        (by simpa [boundEnv, setLocal] using getSize) (by simpa [boundEnv, setLocal] using getPrevious)
    obtain ⟨after, afterEnv, afterHeap, afterAddresses, steps, rest, finalView, finalPrivate, finalPrevious, locals, objects⟩ :=
      forward_scan next (by omega) nextHeap root nextAddresses nextView nextPrivate nextEnv globals
        ((nextLocals _ (by simp)).trans (by simpa [boundEnv, setLocal] using getMatrix))
        ((nextLocals _ (by simp)).trans (by simpa [boundEnv, setLocal] using getSize)) nextPrevious
    refine ⟨after, afterEnv, afterHeap, afterAddresses, .next step steps, ?_, finalView, finalPrivate, finalPrevious, ?_, ?_⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, show Calls.loopControl eliminationBody = false from rfl,
        Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using ran.append rest
    · intro name untouched
      have notColumn : name ≠ "column" := fun same => untouched (by simp [same])
      have other : name ∉ ["pivot_row", "pivot_values", "pivot", "row", "j", "factor", "previous"] :=
        fun member => untouched (List.mem_cons_of_mem _ member)
      exact (locals name untouched).trans ((nextLocals name other).trans (by simp [boundEnv, setLocal, notColumn]))
    · intro address before
      exact (objects address before).trans (nextObjects address before)
  · have finished : k = 119 := by omega
    subst k
    exact ⟨state, env, heap, addresses, .refl state,
      BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, view, privateRows, getPrevious,
      fun _ _ => rfl, fun _ _ => rfl⟩
termination_by 119 - k
decreasing_by omega

theorem forward_for {sides : Nat} (positive : 0 < sides)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View (Monopoly.initialElimination sides positive).matrix heap root addresses)
    (privateRows : ∀ row, root < addresses row) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getSize : env "size" = some (.scalar (.int 120)))
    (getPrevious : env "previous" = some (.scalar (.int 1))) :
    ∃ (after : Bareiss.State (Monopoly.augmentedMatrix sides) 119) (afterEnv : HeapCalls.Env) (afterHeap : Mutable.Heap) (afterAddresses : Fin 120 → Nat),
      Bareiss.Runs (Monopoly.initialElimination sides positive) after ∧
      BlockContinues program env ((executableBody definition.2).drop 4 |>.take 1) heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View after.matrix afterHeap root afterAddresses ∧
      (∀ row, root < afterAddresses row) ∧
      afterEnv "previous" = some (.scalar (.int after.previous)) ∧
      (∀ name, name ∉ ["column", "pivot_row", "pivot_values", "pivot", "row", "j", "factor", "previous"] → afterEnv name = env name) ∧
      ∀ address, address < root → afterHeap[address]? = heap[address]? := by
  obtain ⟨after, afterEnv, afterHeap, afterAddresses, steps, ran, finalView, finalPrivate, finalPrevious, locals, objects⟩ :=
    forward_scan (Monopoly.initialElimination sides positive) (by decide) heap root addresses view privateRows env globals
      getMatrix getSize (by simpa only [Monopoly.initialElimination, Bareiss.State.previous, Bareiss.initial, BareissInvariant.initial_previous] using getPrevious)
  refine ⟨after, afterEnv, afterHeap, afterAddresses, steps, ?_, finalView, finalPrivate, finalPrevious, locals, objects⟩
  exact BlockContinues.for_range (.cons (Evaluates.binary (.lookup getSize) Evaluates.constant rfl) .nil) rfl ran

end solvers.«84».stationary_weights
