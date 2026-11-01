import «.formalsw».solvers.«91.py.ast»
import ProjectEulerSolutions.Lib.LatticeOptimized
import lang.py.FormalSW.Calls

namespace solvers.«91»
open FormalSW.PyAst Calls ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false

abbrev program := Solvers.P91Ast.module
abbrev number (n : Nat) : Constant := .int n

def code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
def outerBody : List Stmt := match (code[2]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
def innerBody : List Stmt := match (outerBody[0]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
def interiorCode : List Stmt := match (innerBody[1]? : Option Stmt) with | some (.ifThen _ _ body) => body | _ => []

structure Ready (env : ScalarEnv) (n x total : Nat) : Prop where
  bound : env "n" = some (number n)
  first : env "x" = some (number x)
  count : env "total" = some (number total)

theorem gcd_result (x y : Nat) (env : ScalarEnv)
    (getX : env "x" = some (number x)) (getY : env "y" = some (number y)) :
    Evaluates program env (.call (.name "gcd") [.name "x", .name "y"]) (number (x.gcd y)) := by
  apply Evaluates.library_call (.cons (Evaluates.lookup getX) (.cons (Evaluates.lookup getY) .nil)) rfl
  change some (Math.gcdCall [.int x, .int y]) = _
  simpa only [Int.gcd_natCast_natCast] using congrArg some (Math.gcdCall_pair x y)

theorem floor_result (left right : Expr) (a b : Nat) (env : ScalarEnv) (positive : 0 < b)
    (leftValue : Evaluates program env left (number a)) (rightValue : Evaluates program env right (number b)) :
    Evaluates program env (.binOp left .floorDiv right) (number (a / b)) := by
  apply Evaluates.binary leftValue rightValue
  simp [scalarBinary, number, positive.ne']

theorem sub_result (left right : Expr) (a b : Nat) (env : ScalarEnv) (bounded : b ≤ a)
    (leftValue : Evaluates program env left (number a)) (rightValue : Evaluates program env right (number b)) :
    Evaluates program env (.binOp left .sub right) (number (a - b)) := by
  exact Evaluates.binary leftValue rightValue (by simp [scalarBinary, number, Int.natCast_sub bounded])

theorem minimum_pair (a b : Int) :
    positionalMinimum [.int a, .int b] = some (.ok (.int (min a b))) := by
  by_cases smaller : b < a
  · simp [positionalMinimum, minimumFrom, scalarCompare, evalCmpOp, smaller, min_eq_right smaller.le]
  · simp [positionalMinimum, minimumFrom, scalarCompare, evalCmpOp, smaller, min_eq_left (le_of_not_gt smaller)]

theorem min_result (left right : Expr) (a b : Nat) (env : ScalarEnv)
    (leftValue : Evaluates program env left (number a)) (rightValue : Evaluates program env right (number b)) :
    Evaluates program env (.builtinCall .min [left, right]) (number (min a b)) := by
  apply Evaluates.builtin (.cons leftValue (.cons rightValue .nil))
  simpa only [number, Nat.cast_min] using minimum_pair a b

theorem zero_test (name : String) (value : Nat) (env : ScalarEnv)
    (found : env name = some (number value)) :
    Evaluates program env (.compare (.name name) [(.eq, .constant (.int 0))]) (.bool (decide (value = 0))) :=
  Evaluates.compare (Evaluates.lookup found) Evaluates.constant (by simp [scalarCompare, evalCmpOp])

theorem origin_test (x y : Nat) (env : ScalarEnv)
    (getX : env "x" = some (number x)) (getY : env "y" = some (number y)) :
    Evaluates program env (.boolOp .and [
      .compare (.name "x") [(.eq, .constant (.int 0))],
      .compare (.name "y") [(.eq, .constant (.int 0))]]) (.bool (decide (x = 0 ∧ y = 0))) := by
  by_cases zero : x = 0
  · apply Evaluates.boolean_continue (by simpa only [zero, decide_true] using zero_test "x" x env getX) rfl
    simpa only [zero, true_and] using Evaluates.boolean_last (zero_test "y" y env getY)
  · simpa only [zero, false_and, decide_false] using
      Evaluates.boolean_stop (op := BoolOp.and) (by simpa only [zero, decide_false] using zero_test "x" x env getX) rfl

theorem axis_test (x y : Nat) (env : ScalarEnv)
    (getX : env "x" = some (number x)) (getY : env "y" = some (number y)) :
    Evaluates program env (.boolOp .or [
      .compare (.name "x") [(.eq, .constant (.int 0))],
      .compare (.name "y") [(.eq, .constant (.int 0))]]) (.bool (decide (x = 0 ∨ y = 0))) := by
  by_cases zero : x = 0
  · simpa only [zero, true_or, decide_true] using
      Evaluates.boolean_stop (op := BoolOp.or) (by simpa only [zero, decide_true] using zero_test "x" x env getX) rfl
  · apply Evaluates.boolean_continue (by simpa only [zero, decide_false] using zero_test "x" x env getX) rfl
    simpa only [zero, false_or] using Evaluates.boolean_last (zero_test "y" y env getY)

/-- Every divisor used by the interior branch is positive, and the assignments realize both exact step bounds. -/
theorem interior (n x y total : Nat) (env : ScalarEnv) (ready : Ready env n x total)
    (getY : env "y" = some (number y)) (positiveX : 0 < x) (positiveY : 0 < y)
    (xBound : x ≤ n) (yBound : y ≤ n) (following : List Stmt) (answer : Constant)
    (rest : ∀ after, Ready after n x (total + upperSteps n x y + lowerSteps n x y) →
      BlockReturns program after following answer) :
    BlockReturns program env (interiorCode ++ following) answer := by
  have gPositive := Nat.gcd_pos_of_pos_left y positiveX
  have dxPositive : 0 < y / x.gcd y := by simpa [Nat.gcd_comm] using divided_gcd_positive y x positiveY
  have dyPositive := divided_gcd_positive x y positiveX
  apply BlockReturns.assign (gcd_result x y env ready.first getY)
  apply BlockReturns.assign (value := number (y / x.gcd y))
  · exact floor_result _ _ y (x.gcd y) _ gPositive
      (Evaluates.lookup (by simpa [ScalarEnv.set] using getY)) (Evaluates.lookup rfl)
  apply BlockReturns.assign (value := number (x / x.gcd y))
  · exact floor_result _ _ x (x.gcd y) _ gPositive
      (Evaluates.lookup (by simpa [ScalarEnv.set] using ready.first)) (Evaluates.lookup rfl)
  apply BlockReturns.assign (value := number (upperSteps n x y))
  · apply min_result
    · exact floor_result _ _ x (y / x.gcd y) _ dxPositive
        (Evaluates.lookup (by simpa [ScalarEnv.set] using ready.first)) (Evaluates.lookup rfl)
    · apply floor_result _ _ (n - y) (x / x.gcd y) _ dyPositive
      · exact sub_result _ _ n y _ yBound
          (Evaluates.lookup (by simpa [ScalarEnv.set] using ready.bound))
          (Evaluates.lookup (by simpa [ScalarEnv.set] using getY))
      · exact Evaluates.lookup rfl
  apply BlockReturns.assign (value := number (lowerSteps n x y))
  · apply min_result
    · apply floor_result _ _ (n - x) (y / x.gcd y) _ dxPositive
      · exact sub_result _ _ n x _ xBound
          (Evaluates.lookup (by simpa [ScalarEnv.set] using ready.bound))
          (Evaluates.lookup (by simpa [ScalarEnv.set] using ready.first))
      · exact Evaluates.lookup rfl
    · exact floor_result _ _ y (x / x.gcd y) _ dyPositive
        (Evaluates.lookup (by simpa [ScalarEnv.set] using getY)) (Evaluates.lookup rfl)
  apply BlockReturns.augAssign (value := number (total + upperSteps n x y + lowerSteps n x y))
  · apply Evaluates.binary (Evaluates.lookup (by simpa [ScalarEnv.set] using ready.count))
      (Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup rfl) rfl)
    simp [scalarBinary, number, Nat.add_assoc]
  · apply rest
    constructor
    · simpa [ScalarEnv.set] using ready.bound
    · simpa [ScalarEnv.set] using ready.first
    · rfl

end solvers.«91»
