import «.formalsw».solvers.«73.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p73.conditions.source»

/-! Python floor division and clamping implement the complete symbolic numerator bounds. -/
namespace solvers.«73»
open FormalSW.PyAst Calls ProjectEulerStatements.FractionIntervalCount
open data.«project-euler-statements».data.py.project_euler_statements.p73
  (Captured values helperArgs naiveArgs)
set_option Elab.async false

abbrev program := Solvers.P73Ast.module
abbrev number (n : Int) : Constant := .int n
abbrev lowerExpression : Expr := .builtinCall .max
  [.constant (.int 1), .binOp
    (.binOp (.binOp (.name "lower_num") .mult (.name "d")) .floorDiv (.name "lower_den"))
    .add (.constant (.int 1))]
abbrev upperExpression : Expr := .builtinCall .min
  [.binOp (.name "d") .sub (.constant (.int 1)),
   .binOp (.binOp (.binOp (.name "upper_num") .mult (.name "d")) .sub (.constant (.int 1)))
     .floorDiv (.name "upper_den")]

theorem maximum_pair (a b : Int) :
    positionalMaximum [number a, number b] = some (.ok (number (max a b))) := by
  by_cases larger : a < b
  · simp [positionalMaximum, maximumFrom, scalarCompare, evalCmpOp, larger, max_eq_right larger.le]
  · simp [positionalMaximum, maximumFrom, scalarCompare, evalCmpOp, larger,
      max_eq_left (le_of_not_gt larger)]

theorem minimum_pair (a b : Int) :
    positionalMinimum [number a, number b] = some (.ok (number (min a b))) := by
  by_cases smaller : b < a
  · simp [positionalMinimum, minimumFrom, scalarCompare, evalCmpOp, smaller,
      min_eq_right smaller.le]
  · simp [positionalMinimum, minimumFrom, scalarCompare, evalCmpOp, smaller,
      min_eq_left (le_of_not_gt smaller)]

theorem floor_result (left right : Expr) (a b : Int) (env : ScalarEnv) (positive : 0 < b)
    (leftValue : Evaluates program env left (number a))
    (rightValue : Evaluates program env right (number b)) :
    Evaluates program env (.binOp left .floorDiv right) (number (a / b)) :=
  Evaluates.binary leftValue rightValue (by simp [scalarBinary, positive.ne', positive])

theorem lower_result (p : Bounds) (d : Int) (env : ScalarEnv) (positive : Positive p)
    (captured : Captured p env) (getD : env "d" = some (number d)) :
    Evaluates program env lowerExpression (number (lower p d)) := by
  apply Evaluates.builtin (values := [number 1, number (p.lowerNum * d / p.lowerDen + 1)])
  · apply Evaluates.Arguments.cons Evaluates.constant
    refine .cons ?_ .nil
    exact Evaluates.binary (av := number (p.lowerNum * d / p.lowerDen)) (bv := number 1)
      (floor_result _ _ (p.lowerNum * d) p.lowerDen env positive.1
        (Evaluates.binary (Evaluates.lookup captured.lowerNum) (Evaluates.lookup getD) rfl)
        (Evaluates.lookup captured.lowerDen)) Evaluates.constant rfl
  · exact maximum_pair 1 _

theorem upper_result (p : Bounds) (d : Int) (env : ScalarEnv) (positive : Positive p)
    (captured : Captured p env) (getD : env "d" = some (number d)) :
    Evaluates program env upperExpression (number (upper p d)) := by
  apply Evaluates.builtin
    (values := [number (d - 1), number ((p.upperNum * d - 1) / p.upperDen)])
  · apply Evaluates.Arguments.cons
      (Evaluates.binary (Evaluates.lookup getD) Evaluates.constant rfl)
    refine .cons ?_ .nil
    exact floor_result _ _ (p.upperNum * d - 1) p.upperDen env positive.2
      (Evaluates.binary (av := number (p.upperNum * d)) (bv := number 1)
        (Evaluates.binary (Evaluates.lookup captured.upperNum) (Evaluates.lookup getD) rfl)
        Evaluates.constant rfl) (Evaluates.lookup captured.upperDen)
  · exact minimum_pair (d - 1) _

abbrev gcdCondition : Expr := .compare (.call (.name "gcd") [.name "n", .name "d"])
  [(.eq, .constant (.int 1))]

theorem gcd_result (n d : Int) (env : ScalarEnv)
    (getN : env "n" = some (number n)) (getD : env "d" = some (number d)) :
    Evaluates program env gcdCondition (.bool (decide (Int.gcd n d = 1))) := by
  apply Evaluates.compare (av := .int (Int.gcd n d)) (bv := .int 1)
  · apply Evaluates.library_call
      (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getD) .nil)) rfl
    change some (Math.gcdCall [.int n, .int d]) = _
    simpa using congrArg some (Math.gcdCall_pair n d)
  · exact Evaluates.constant
  · simp [scalarCompare, evalCmpOp]

end solvers.«73»
