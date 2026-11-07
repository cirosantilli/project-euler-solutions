import solvers.«84.py.lean.d».StationaryColumnLoop

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

/-- One complete source iteration advances the mathematical invariant and saves the new divisor. -/
theorem elimination_iteration {sides k : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) k)
    (unfinished : k < 120) (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View state.matrix heap root addresses)
    (privateRows : ∀ row, root < addresses row)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getColumn : env "column" = some (.scalar (.int k)))
    (getSize : env "size" = some (.scalar (.int 120)))
    (getPrevious : env "previous" = some (.scalar (.int state.previous))) :
    ∃ (after : Bareiss.State (Monopoly.augmentedMatrix sides) (k + 1)) (afterEnv : HeapCalls.Env) (afterHeap : Mutable.Heap) (afterAddresses : Fin 120 → Nat),
      Bareiss.Step state after ∧
      BlockContinues program env eliminationBody heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View after.matrix afterHeap root afterAddresses ∧
      (∀ row, root < afterAddresses row) ∧
      afterEnv "previous" = some (.scalar (.int after.previous)) ∧
      (∀ name, name ∉ ["pivot_row", "pivot_values", "pivot", "row", "j", "factor", "previous"] → afterEnv name = env name) ∧
      ∀ address, address < root → afterHeap[address]? = heap[address]? := by
  obtain ⟨candidate, bound, nonzero, prefixRun, swappedView⟩ := pivot_prefix state unfinished heap root addresses view
    env globals getMatrix getColumn getSize
  let swapped := state.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound
  let pivotEnv := swappedEnv (setLocal env "pivot_row" (.scalar (.int candidate.val))) addresses state.matrix ⟨k, unfinished⟩ candidate
  let pivotAddresses := addresses ∘ Equiv.swap ⟨k, unfinished⟩ candidate
  have pivotMatrix : pivotEnv "matrix" = some (.ref root) := by simpa [pivotEnv, swappedEnv, setLocal] using getMatrix
  have pivotColumn : pivotEnv "column" = some (.scalar (.int k)) := by simpa [pivotEnv, swappedEnv, setLocal] using getColumn
  have pivotSize : pivotEnv "size" = some (.scalar (.int 120)) := by simpa [pivotEnv, swappedEnv, setLocal] using getSize
  have pivotValues : pivotEnv "pivot_values" = some (.ref (pivotAddresses ⟨k, unfinished⟩)) := by
    simp [pivotEnv, swappedEnv, pivotAddresses, setLocal]
  have pivotValue : pivotEnv "pivot" = some (.scalar (.int (swapped.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩))) := by
    simp [pivotEnv, swappedEnv, swapped, Bareiss.State.swap, IntegerRowSwap.swap, setLocal]
  have pivotPrevious : pivotEnv "previous" = some (.scalar (.int swapped.previous)) := by
    simpa [pivotEnv, swappedEnv, swapped, Bareiss.State.swap, Bareiss.State.previous, setLocal] using getPrevious
  obtain ⟨rowEnv, rowHeap, rowRun, finalView, locals, objects⟩ := column_result swapped unfinished nonzero
    (swappedHeap heap root addresses ⟨k, unfinished⟩ candidate) root pivotAddresses swappedView pivotEnv globals
    pivotMatrix pivotColumn pivotSize pivotValues pivotValue pivotPrevious
  have finalPivot : rowEnv "pivot" = some (.scalar (.int (swapped.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩))) :=
    (locals _ (by decide) (by decide) (by decide)).trans pivotValue
  let after := swapped.advance unfinished nonzero
  let afterEnv := setLocal rowEnv "previous" (.scalar (.int (swapped.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩)))
  refine ⟨after, afterEnv, rowHeap, pivotAddresses, ⟨unfinished, candidate, bound, nonzero, rfl⟩,
    ?_, finalView, fun row => privateRows _, ?_, ?_, ?_⟩
  · exact prefixRun.append (rowRun.append (BlockContinues.assign (name := "previous") (.lookup finalPivot)))
  · simp [afterEnv, setLocal, after, Bareiss.State.advance_previous]
  · intro name untouched
    simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at untouched
    rcases untouched with ⟨notChoice, notValues, notPivot, notRow, notIndex, notFactor, notPrevious⟩
    simp only [afterEnv, setLocal, if_neg notPrevious]
    rw [locals name notRow notIndex notFactor]
    simp [pivotEnv, swappedEnv, setLocal, notChoice, notValues, notPivot]
  · intro address before
    have untouched := objects address (fun row => by
      have above : root < pivotAddresses row := privateRows _
      omega)
    exact untouched.trans (List.getElem?_set_ne (by omega))

end solvers.«84».stationary_weights
