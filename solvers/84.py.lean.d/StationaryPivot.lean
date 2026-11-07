import solvers.«84.py.lean.d».StationarySwap
import lang.py.FormalSW.NextCalls

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def pivotCondition : Expr := .subscript (.subscript (.name "matrix") (.name "row")) (.name "column")
def pivotExpression : Expr :=
  .consumeGenerator (nextConsumer (some (.constant .none))) (.name "row")
    [(.name "row", .call (.name "range") [.name "column", .name "size"], [pivotCondition])]

theorem pivot_source : eliminationBody.head? = some (.assign [.name "pivot_row"] pivotExpression) := rfl

theorem pivot_scan (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (column candidate : Fin 120) (nonzero : matrix candidate column.castSucc ≠ 0)
    (first : Nat) (bound : first ≤ candidate.val)
    (zeros : ∀ row : Fin 120, first ≤ row.val → row.val < candidate.val → matrix row column.castSucc = 0)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getColumn : env "column" = some (.scalar (.int column.val))) (following : List Stmt) :
    BlockReturns program env (.rangeStepLoop "row" first 120 1
      [.ifThen pivotCondition [.return (.name "row")] []] [] :: following)
      heap (.scalar (.int candidate.val)) heap globals := by
  have inside : first < 120 := lt_of_le_of_lt bound candidate.isLt
  let row : Fin 120 := ⟨first, inside⟩
  let boundEnv := setLocal env "row" (.scalar (.int first))
  have boundMatrix : boundEnv "matrix" = some (.ref root) := by simpa [boundEnv, setLocal] using getMatrix
  have boundColumn : boundEnv "column" = some (.scalar (.int column.val)) := by simpa [boundEnv, setLocal] using getColumn
  have tested : Evaluates program boundEnv pivotCondition heap (.scalar (.int (matrix row column.castSucc))) heap globals :=
    view.entry_result program boundEnv globals _ _ row column.castSucc
      (view.row_result program boundEnv globals _ _ row (.lookup boundMatrix) (.lookup rfl) (by intros; simp))
      (.lookup boundColumn) (by intros; simp)
  apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
  by_cases found : first = candidate.val
  · have same : row = candidate := Fin.ext found
    apply BlockReturns.branch (flag := true) tested
      (by simp [Mutable.truth, scalarTruth, same, nonzero])
    exact BlockReturns.return_value (.lookup (by simp [boundEnv, setLocal, found]))
  · have before : first < candidate.val := by omega
    have zero := zeros row (Nat.le_refl _) before
    apply BlockReturns.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, zero])
    have rest := pivot_scan matrix heap root addresses view column candidate nonzero (first + 1) (by omega)
      (fun row lower upper => zeros row (by omega) upper) boundEnv globals boundMatrix boundColumn following
    simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest
termination_by candidate.val - first
decreasing_by omega

/-- A nonzero pivot is found by the actual filtered range generator, without exhausting the range. -/
theorem pivot_result (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (column candidate : Fin 120) (bound : column.val ≤ candidate.val)
    (nonzero : matrix candidate column.castSucc ≠ 0)
    (zeros : ∀ row : Fin 120, column.val ≤ row.val → row.val < candidate.val → matrix row column.castSucc = 0)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getColumn : env "column" = some (.scalar (.int column.val)))
    (getSize : env "size" = some (.scalar (.int 120))) :
    Evaluates program env pivotExpression heap (.scalar (.int candidate.val)) heap globals := by
  apply Evaluates.next_generator_range (.cons (.lookup getColumn) (.cons (.lookup getSize) .nil)) rfl Evaluates.constant
  exact pivot_scan matrix heap root addresses view column candidate nonzero column.val bound zeros _ globals
    (by simpa [nextEnv, nextDefault, comprehensionTargetsEnv, targetLocals, setLocal] using getMatrix)
    (by simpa [nextEnv, nextDefault, comprehensionTargetsEnv, targetLocals, setLocal] using getColumn) _

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
