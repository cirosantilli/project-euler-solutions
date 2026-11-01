import solvers.«80.py.lean.d».Prefix

namespace solvers.«80».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.SquareRootDigitCalls
open ProjectEulerStatements
set_option Elab.async false
private abbrev program := Solvers.P80Ast.module
variable {globals : Mutable.Globals}

def loopBody : List Stmt :=
  [.augAssign (.name "total") .add (.call (.name "sqrt_digits_sum") [.name "n", .name "digits"])]
private abbrev finish : Stmt := .return (.name "total")

theorem loop_result (stop first count digits total : Nat) (endpoint : first + count = stop)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getDigits : env "digits" = some (number digits))
    (getTotal : env "total" = some (number total)) :
    BlockReturns program env [.rangeStepLoop "n" first stop 1 loopBody [], finish] heap
      (number ((List.range' first count).foldl (fun acc n => acc + P80.sqrtDigitsSum n digits) total)) heap globals := by
  induction count generalizing first total env with
  | zero =>
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | succ count ih =>
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockReturns.augAssign (afterOperation := heap) (value := number (total + P80.sqrtDigitsSum first digits))
        (by simpa [setLocal] using getTotal)
      · exact Evaluates.call
          (.cons (Evaluates.lookup (by simp [setLocal]))
            (.cons (Evaluates.lookup (by simpa [setLocal] using getDigits)) .nil))
          (sqrt_digits_sum.result first digits heap globals)
      · simp [inplace, Mutable.binary, scalarBinary, number]
      · simpa only [List.range'_succ, List.foldl_cons, loopBody, Calls.loopContinuation,
          show Calls.loopControl loopBody = false from rfl, Bool.false_eq_true, if_false,
          Int.natCast_add, Int.natCast_one] using
          ih (first + 1) (total + P80.sqrtDigitsSum first digits) (by omega)
            (setLocal (setLocal env "n" (number first)) "total" (number (total + P80.sqrtDigitsSum first digits)))
            (by simpa [setLocal] using getDigits) (by simp [setLocal])

theorem result (limit digits : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "solve" [number limit, number digits] heap
      (number (P80.naive limit digits)) heap globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.for_range (start := 1) (stop := (limit + 1 : Nat)) (stride := 1)
    (.cons Evaluates.constant (.cons (Evaluates.binary (value := number (limit + 1)) (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
  rw [SquareRootDigits.skip_zero]
  exact loop_result (limit + 1) 1 limit digits 0 (by omega) heap _
    (by cases globals <;> rfl) (by simp [setLocal])

theorem default_result (heap : Mutable.Heap) :
    ReturnsValue program "solve" [] heap (number (P80.naive 100 100)) heap :=
  ReturnsValue.with_defaults rfl rfl rfl (result 100 100 heap)

end solvers.«80».solve
