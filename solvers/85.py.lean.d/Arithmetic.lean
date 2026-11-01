import «.formalsw».solvers.«85.py.ast»
import ProjectEulerSolutions.Lib.RectangleApproximation
import ProjectEulerSolutions.Lib.SquareRootDigitCalls

/-! Arithmetic and candidate expressions for the optimized rectangle search. -/
namespace solvers.«85»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangleApproximation
open ProjectEulerSolutions.TriPentHex (triangular)
set_option Elab.async false

abbrev program := Solvers.P85Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

private theorem add {module : Module} {env : HeapCalls.Env} {heap : Mutable.Heap}
    {globals : Mutable.Globals} {left right : Expr} {a b : Nat}
    (ha : Evaluates module env left heap (number a) heap globals)
    (hb : Evaluates module env right heap (number b) heap globals) :
    Evaluates module env (.binOp left .add right) heap (number (a + b)) heap globals :=
  Evaluates.binary ha hb (by simp [Mutable.binary, scalarBinary, number])

private theorem multiply {module : Module} {env : HeapCalls.Env} {heap : Mutable.Heap}
    {globals : Mutable.Globals} {left right : Expr} {a b : Nat}
    (ha : Evaluates module env left heap (number a) heap globals)
    (hb : Evaluates module env right heap (number b) heap globals) :
    Evaluates module env (.binOp left .mult right) heap (number (a * b)) heap globals :=
  Evaluates.binary ha hb (by simp [Mutable.binary, scalarBinary, number])

private theorem divide {module : Module} {env : HeapCalls.Env} {heap : Mutable.Heap}
    {globals : Mutable.Globals} {left right : Expr} {a b : Nat} (positive : 0 < b)
    (ha : Evaluates module env left heap (number a) heap globals)
    (hb : Evaluates module env right heap (number b) heap globals) :
    Evaluates module env (.binOp left .floorDiv right) heap (number (a / b)) heap globals := by
  apply Evaluates.binary ha hb
  simp [Mutable.binary, scalarBinary, number, Int.natCast_ediv, positive.ne', positive]

/-- The optimized helper computes the rectangle formula without modifying caller state. -/
theorem rectangle_count.result (m n : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "rectangle_count" [number m, number n] heap
      (number (count m n)) heap globals := by
  rw [count_formula]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply divide (by omega)
  · apply multiply
    · apply multiply
      · exact multiply (Evaluates.lookup (by cases globals <;> rfl))
          (add (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant)
      · exact Evaluates.lookup (by cases globals <;> rfl)
    · exact add (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
  · exact Evaluates.constant

private theorem maximum_operation (a b : Int) (heap : Mutable.Heap) :
    primitive (.builtin .max) [.scalar (.int a), .scalar (.int b)] heap =
      some (.ok (.scalar (.int (max a b))), heap) := by
  simp [primitive, scalarResult, Calls.positionalMaximum,
    Calls.maximumFrom, scalarCompare, evalCmpOp, max_def]
  split <;> split <;> simp_all <;> omega

/-- The signed Python quotient and zero clamp produce the natural row-search bound. -/
theorem quotient_evaluate (target : Int) {m : Nat} (positive : 0 < m)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getTarget : env "target" = some (.scalar (.int target)))
    (getVertical : env "vertical" = some (number (triangular m))) :
    Evaluates program env (.builtinCall .max [.constant (.int 0),
      .binOp (.name "target") .floorDiv (.name "vertical")]) heap
      (number (target.toNat / triangular m)) heap globals := by
  have verticalPositive := triangular_positive positive
  apply Evaluates.builtin (middle := heap)
    (values := [.scalar (.int 0), .scalar (.int (target / (triangular m : Int)))])
  · apply Evaluates.Arguments.cons Evaluates.constant
    refine .cons ?_ .nil
    exact Evaluates.binary (Evaluates.lookup getTarget) (Evaluates.lookup getVertical)
      (by simp [Mutable.binary, scalarBinary, verticalPositive.ne', verticalPositive])
  · simpa only [clamped_quotient target (triangular_positive positive)] using
      maximum_operation 0 (target / (triangular m : Int)) heap

/-- The actual integer-square-root expression evaluates to the complete lower candidate. -/
theorem below_evaluate (target : Int) (m : Nat)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getM : env "m" = some (number m))
    (getQuotient : env "quotient" = some (number (target.toNat / triangular m))) :
    Evaluates program env (.builtinCall .max [.name "m",
      .binOp (.binOp (.call (.name "isqrt")
        [.binOp (.constant (.int 1)) .add
          (.binOp (.constant (.int 8)) .mult (.name "quotient"))])
        .sub (.constant (.int 1))) .floorDiv (.constant (.int 2))]) heap
      (number (rowBelow target m)) heap globals := by
  let q := target.toNat / triangular m
  have rootPositive : 1 ≤ Nat.sqrt (1 + 8 * q) := Nat.sqrt_pos.mpr (by omega)
  apply Evaluates.builtin (middle := heap) (values := [number m, number (inverseTriangular q)])
  · apply Evaluates.Arguments.cons (Evaluates.lookup getM)
    refine .cons ?_ .nil
    apply divide (by omega)
    · apply Evaluates.binary (av := number (Nat.sqrt (1 + 8 * q))) (bv := number 1)
      · apply ProjectEulerSolutions.SquareRootDigitCalls.isqrt_call
        · exact add Evaluates.constant (multiply Evaluates.constant (Evaluates.lookup getQuotient))
        · rfl
        · rfl
      · exact Evaluates.constant
      · simp [Mutable.binary, scalarBinary, number, Int.ofNat_sub rootPositive]
    · exact Evaluates.constant
  · simpa [number, rowBelow, rowIndex, q] using
      maximum_operation (m : Int) (inverseTriangular q : Int) heap

end solvers.«85»
