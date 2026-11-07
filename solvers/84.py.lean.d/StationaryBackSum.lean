import solvers.«84.py.lean.d».StationaryBackBuffer
import lang.py.FormalSW.SumCalls

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open scoped BigOperators
set_option Elab.async false

def upperProductSum (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int) (row : Fin 120) (first : Nat) : Int :=
  ∑ column ∈ Finset.univ.filter (fun column : Fin 120 => first ≤ column.val), matrix row column.castSucc * values column

theorem upperProductSum_step (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (row column : Fin 120) :
    upperProductSum matrix values row column.val = matrix row column.castSucc * values column +
      upperProductSum matrix values row (column.val + 1) := by
  have domain : Finset.univ.filter (fun index : Fin 120 => column.val ≤ index.val) =
      insert column (Finset.univ.filter (fun index : Fin 120 => column.val + 1 ≤ index.val)) := by
    ext index
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Fin.ext_iff]
    omega
  rw [upperProductSum, domain, Finset.sum_insert (by simp)]
  rfl

theorem upperProductSum_end (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int) (row : Fin 120) :
    upperProductSum matrix values row 120 = 0 := by
  apply Finset.sum_eq_zero
  intro column member
  have above := (Finset.mem_filter.mp member).2
  have below := column.isLt
  omega

def backProduct : Expr := .binOp (.subscript (.subscript (.name "matrix") (.name "row")) (.name "j")) .mult
  (.subscript (.name "weights") (.name "j"))

theorem product_value (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView matrix values heap root addresses weights) (row column : Fin 120)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getWeights : env "weights" = some (.ref weights))
    (getRow : env "row" = some (.scalar (.int row.val))) (getColumn : env "j" = some (.scalar (.int column.val))) :
    Evaluates program env backProduct heap (.scalar (.int (matrix row column.castSucc * values column))) heap globals := by
  apply Evaluates.binary
    (view.matrixView.entry_result program env globals _ _ row column.castSucc
      (view.matrixView.row_result program env globals _ _ row (.lookup getMatrix) (.lookup getRow) (by intros; simp))
      (.lookup getColumn) (by intros; simp))
    (Evaluates.subscript (by intros; simp) (.lookup getWeights) (.lookup getColumn) (view.read column)) rfl

theorem back_sum_scan (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView matrix values heap root addresses weights) (row : Fin 120) (first : Nat) (bound : first ≤ 120)
    (total : Int) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getWeights : env "weights" = some (.ref weights))
    (getRow : env "row" = some (.scalar (.int row.val)))
    (getTotal : env sumAccumulator = some (.scalar (.int total))) :
    BlockReturns program env [.rangeStepLoop "j" first 120 1 (sumBody backProduct []) [], .return (.name sumAccumulator)]
      heap (.scalar (.int (total + upperProductSum matrix values row first))) heap globals := by
  by_cases active : first < 120
  · let column : Fin 120 := ⟨first, active⟩
    let boundEnv := setLocal env "j" (.scalar (.int first))
    have boundMatrix : boundEnv "matrix" = some (.ref root) := by simpa [boundEnv, setLocal] using getMatrix
    have boundWeights : boundEnv "weights" = some (.ref weights) := by simpa [boundEnv, setLocal] using getWeights
    have boundRow : boundEnv "row" = some (.scalar (.int row.val)) := by simpa [boundEnv, setLocal] using getRow
    have boundTotal : boundEnv sumAccumulator = some (.scalar (.int total)) := by simpa [boundEnv, setLocal, sumAccumulator] using getTotal
    have product := product_value matrix values heap root weights addresses view row column boundEnv globals boundMatrix boundWeights boundRow rfl
    have added := BlockContinues.assign (name := sumAccumulator) (Evaluates.binary (op := .add) (.lookup boundTotal) product rfl)
    have rest := back_sum_scan matrix values heap root weights addresses view row (first + 1) (by omega)
      (total + matrix row column.castSucc * values column)
      (setLocal boundEnv sumAccumulator (.scalar (.int (total + matrix row column.castSucc * values column)))) globals
      (by simpa [setLocal, sumAccumulator] using boundMatrix)
      (by simpa [setLocal, sumAccumulator] using boundWeights)
      (by simpa [setLocal, sumAccumulator] using boundRow) (by simp [setLocal])
    apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
    rw [show first = column.val from rfl, upperProductSum_step, ← add_assoc]
    simpa only [sumBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or,
      Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using added.returns rest
  · have finished : first = 120 := by omega
    subst first
    rw [upperProductSum_end, add_zero]
    exact BlockReturns.range_done (by simp [Calls.rangeContinues]) (BlockReturns.return_value (.lookup getTotal))
termination_by 120 - first
decreasing_by omega

def backSum : Expr := .sumRange "j" backProduct
  [.binOp (.name "row") .add (.constant (.int 1)), .name "size"] []

theorem back_sum_result (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView matrix values heap root addresses weights) (row : Fin 120)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getWeights : env "weights" = some (.ref weights))
    (getRow : env "row" = some (.scalar (.int row.val))) (getSize : env "size" = some (.scalar (.int 120))) :
    Evaluates program env backSum heap (.scalar (.int (upperProductSum matrix values row (row.val + 1)))) heap globals := by
  apply Evaluates.sum_range
  apply Evaluates.sum_generator_range (middle := heap) (afterStart := heap) (values := [.scalar (.int (row.val + 1 : Nat)), .scalar (.int 120)])
    (.cons (Evaluates.binary (.lookup getRow) Evaluates.constant (by simp [Mutable.binary, scalarBinary]))
      (.cons (.lookup getSize) .nil)) rfl Evaluates.constant rfl
  simpa only [comprehensionFilters, zero_add] using back_sum_scan matrix values heap root weights addresses view row
    (row.val + 1) (Nat.succ_le_of_lt row.isLt) 0 _ globals
    (by simpa [setLocal, sumAccumulator, comprehensionEnv] using getMatrix)
    (by simpa [setLocal, sumAccumulator, comprehensionEnv] using getWeights)
    (by simpa [setLocal, sumAccumulator, comprehensionEnv] using getRow) (by simp [setLocal])

end solvers.«84».stationary_weights
