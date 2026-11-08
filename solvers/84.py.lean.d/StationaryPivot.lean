import solvers.«84.py.lean.d».StationarySwap
import ProjectEulerSolutions.Lib.MatrixPivotCalls

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

abbrev pivotCondition := ProjectEulerSolutions.MatrixPivotCalls.condition
abbrev pivotExpression := ProjectEulerSolutions.MatrixPivotCalls.expression

theorem pivot_source : eliminationBody.head? = some (.assign [.name "pivot_row"] pivotExpression) := rfl

abbrev pivot_scan := ProjectEulerSolutions.MatrixPivotCalls.scan program
abbrev pivot_result := ProjectEulerSolutions.MatrixPivotCalls.result program

theorem first_pivot {k : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) k) (unfinished : k < 120) :
    ∃ candidate : Fin 120, k ≤ candidate.val ∧ state.matrix candidate ⟨k, by omega⟩ ≠ 0 ∧
      ∀ row : Fin 120, k ≤ row.val → row.val < candidate.val → state.matrix row ⟨k, by omega⟩ = 0 := by
  have existsIndex : ∃ index : Nat, ∃ inside : index < 120,
      k ≤ index ∧ state.matrix ⟨index, inside⟩ ⟨k, by omega⟩ ≠ 0 := by
    obtain ⟨row, lower, nonzero⟩ := state.exists_pivot unfinished
    exact ⟨row.val, row.isLt, lower, nonzero⟩
  obtain ⟨inside, lower, nonzero⟩ := Nat.find_spec existsIndex
  refine ⟨⟨Nat.find existsIndex, inside⟩, lower, nonzero, ?_⟩
  intro row after before
  by_contra nonzero
  exact Nat.find_min existsIndex before ⟨row.isLt, after, nonzero⟩

theorem pivot_guard (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (candidate : Fin 120) (getPivot : env "pivot_row" = some (.scalar (.int candidate.val))) :
    BlockContinues program env ((eliminationBody.drop 1).take 1) heap env heap globals := by
  apply BlockContinues.branch (flag := false) (value := .scalar (.bool false))
    (Evaluates.builtin (.cons (.lookup getPivot) (.cons Evaluates.constant .nil))
      (by simp [HeapCalls.primitive, HeapCalls.identity, HeapCalls.scalarResult, scalarIdentity])) rfl BlockContinues.nil

/-- The first six source statements choose, validate, and install a nonzero pivot. -/
theorem pivot_prefix {sides k : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) k)
    (unfinished : k < 120) (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat)
    (view : MutableIntegerMatrix.View state.matrix heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getColumn : env "column" = some (.scalar (.int k)))
    (getSize : env "size" = some (.scalar (.int 120))) :
    ∃ (candidate : Fin 120) (bound : k ≤ candidate.val),
      let swapped := state.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound
      swapped.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0 ∧
      BlockContinues program env (eliminationBody.take 6) heap
        (swappedEnv (setLocal env "pivot_row" (.scalar (.int candidate.val))) addresses state.matrix ⟨k, unfinished⟩ candidate)
        (swappedHeap heap root addresses ⟨k, unfinished⟩ candidate) globals ∧
      MutableIntegerMatrix.View swapped.matrix (swappedHeap heap root addresses ⟨k, unfinished⟩ candidate)
        root (addresses ∘ Equiv.swap ⟨k, unfinished⟩ candidate) := by
  obtain ⟨candidate, bound, nonzero, zeros⟩ := first_pivot state unfinished
  obtain ⟨swapped, afterView⟩ := swap_result state.matrix heap root addresses view ⟨k, unfinished⟩ candidate
    (setLocal env "pivot_row" (.scalar (.int candidate.val))) globals
    (by simpa [setLocal] using getMatrix) (by simpa [setLocal] using getColumn) rfl
  refine ⟨candidate, bound, ?_, ?_, afterView⟩
  · simpa only [Bareiss.State.swap, IntegerRowSwap.swap, Equiv.swap_apply_left] using nonzero
  · have selected := BlockContinues.assign (name := "pivot_row")
      (pivot_result state.matrix heap root addresses view ⟨k, unfinished⟩ candidate bound nonzero zeros
        env globals getMatrix getColumn getSize)
    have guard := pivot_guard heap (setLocal env "pivot_row" (.scalar (.int candidate.val))) globals candidate rfl
    exact selected.append (guard.append swapped)

end solvers.«84».stationary_weights
