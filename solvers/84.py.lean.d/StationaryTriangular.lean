import solvers.«84.py.lean.d».StationaryForward

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

/-- Matrix construction and the entire source forward-elimination loop reach an invertible triangular system. -/
theorem triangular_result (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (value : Mutable.Value) (view : transitions.View sides heap value)
    (getTransitions : env "transitions" = some value)
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides)))) :
    ∃ (state : Bareiss.State (Monopoly.augmentedMatrix sides) 119)
      (afterEnv : HeapCalls.Env) (afterHeap : Mutable.Heap) (addresses : Fin 120 → Nat),
      Bareiss.Runs (Monopoly.initialElimination sides positive) state ∧
      BlockContinues program env ((executableBody definition.2).take 5) heap afterEnv afterHeap globals ∧
      MutableIntegerMatrix.View state.matrix afterHeap heap.length addresses ∧
      (∀ row, heap.length < addresses row) ∧
      afterEnv "matrix" = some (.ref heap.length) ∧
      afterEnv "size" = some (.scalar (.int 120)) ∧
      ∀ address, address < heap.length → afterHeap[address]? = heap[address]? := by
  obtain ⟨setup, _, extra, prepared⟩ := setup_result sides heap env globals value view getTransitions getDenominator
  let initialEnv := setLocal (setupEnv env heap) "previous" (.scalar (.int 1))
  obtain ⟨state, afterEnv, afterHeap, addresses, steps, ran, finalView, privateRows, _, locals, objects⟩ := forward_for positive
    (preparedHeap heap sides) heap.length (rowAddresses heap) (prepared_mutable_view heap sides positive)
    (rowAddresses_after heap) initialEnv globals (by simp [initialEnv, setupEnv, setLocal])
    (by simp [initialEnv, setupEnv, setLocal]) rfl
  refine ⟨state, afterEnv, afterHeap, addresses, steps, ?_, finalView, privateRows, ?_, ?_, ?_⟩
  · exact setup.append ((BlockContinues.assign (name := "previous") Evaluates.constant).append ran)
  · exact (locals _ (by simp)).trans (by simp [initialEnv, setupEnv, setLocal])
  · exact (locals _ (by simp)).trans (by simp [initialEnv, setupEnv, setLocal])
  · intro address inside
    rw [objects address inside, prepared, List.getElem?_append_left inside]

end solvers.«84».stationary_weights
