import ProjectEulerSolutions.Lib.MutableIntegerMatrixAccess
import lang.py.FormalSW.NextCalls

/-! First-nonzero pivot search through the actual filtered source range generator. -/
namespace ProjectEulerSolutions.MatrixPivotCalls
open FormalSW.PyAst HeapCalls
set_option Elab.async false
variable (module : Module)

def condition : Expr := .subscript (.subscript (.name "matrix") (.name "row")) (.name "column")
def expression : Expr :=
  .consumeGenerator (nextConsumer (some (.constant .none))) (.name "row")
    [(.name "row", .call (.name "range") [.name "column", .name "size"], [condition])]

theorem scan (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (column candidate : Fin 120) (nonzero : matrix candidate column.castSucc ≠ 0)
    (first : Nat) (bound : first ≤ candidate.val)
    (zeros : ∀ row : Fin 120, first ≤ row.val → row.val < candidate.val → matrix row column.castSucc = 0)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getColumn : env "column" = some (.scalar (.int column.val))) (following : List Stmt) :
    BlockReturns module env (.rangeStepLoop "row" first 120 1
      [.ifThen condition [.return (.name "row")] []] [] :: following)
      heap (.scalar (.int candidate.val)) heap globals := by
  have inside : first < 120 := lt_of_le_of_lt bound candidate.isLt
  let row : Fin 120 := ⟨first, inside⟩
  let boundEnv := setLocal env "row" (.scalar (.int first))
  have boundMatrix : boundEnv "matrix" = some (.ref root) := by simpa [boundEnv, setLocal] using getMatrix
  have boundColumn : boundEnv "column" = some (.scalar (.int column.val)) := by simpa [boundEnv, setLocal] using getColumn
  have tested : Evaluates module boundEnv condition heap (.scalar (.int (matrix row column.castSucc))) heap globals :=
    view.entry_result module boundEnv globals _ _ row column.castSucc
      (view.row_result module boundEnv globals _ _ row (.lookup boundMatrix) (.lookup rfl) (by intros; simp))
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
    have rest := scan matrix heap root addresses view column candidate nonzero (first + 1) (by omega)
      (fun row lower upper => zeros row (by omega) upper) boundEnv globals boundMatrix boundColumn following
    simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest
termination_by candidate.val - first
decreasing_by omega

/-- A nonzero pivot is found by the actual filtered range generator, without exhausting the range. -/
theorem result (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses)
    (column candidate : Fin 120) (bound : column.val ≤ candidate.val)
    (nonzero : matrix candidate column.castSucc ≠ 0)
    (zeros : ∀ row : Fin 120, column.val ≤ row.val → row.val < candidate.val → matrix row column.castSucc = 0)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root))
    (getColumn : env "column" = some (.scalar (.int column.val)))
    (getSize : env "size" = some (.scalar (.int 120))) :
    Evaluates module env expression heap (.scalar (.int candidate.val)) heap globals := by
  apply Evaluates.next_generator_range (.cons (.lookup getColumn) (.cons (.lookup getSize) .nil)) rfl Evaluates.constant
  exact scan module matrix heap root addresses view column candidate nonzero column.val bound zeros _ globals
    (by simpa [nextEnv, nextDefault, comprehensionTargetsEnv, targetLocals, setLocal] using getMatrix)
    (by simpa [nextEnv, nextDefault, comprehensionTargetsEnv, targetLocals, setLocal] using getColumn) _

end ProjectEulerSolutions.MatrixPivotCalls
