import «.formalsw».solvers.«86.py.ast»
import ProjectEulerSolutions.Lib.CuboidRoute
import lang.py.FormalSW.Calls

namespace solvers.«86»
open FormalSW.PyAst Calls ProjectEulerSolutions.CuboidRoute
set_option Elab.async false

abbrev program := Solvers.P86Ast.module
abbrev number (n : Nat) : Constant := .int n

abbrev distanceExpression : Expr := .binOp (.binOp (.name "s") .mult (.name "s")) .add (.name "c2")
abbrev lowerExpression : Expr := .builtinCall .max
  [.constant (.int 1), .binOp (.name "s") .sub (.name "c")]
abbrev upperExpression : Expr := .binOp (.name "s") .floorDiv (.constant (.int 2))
abbrev additionExpression : Expr := .binOp (.binOp (.name "hi") .sub (.name "lo")) .add (.constant (.int 1))
abbrev squareCondition : Expr := .compare (.binOp (.name "t") .mult (.name "t")) [(.eq, .name "d")]

theorem distance_evaluate (c s : Nat) (env : ScalarEnv)
    (getS : env "s" = some (number s)) (getC2 : env "c2" = some (number (c ^ 2))) :
    Evaluates program env distanceExpression (number (s ^ 2 + c ^ 2)) := by
  exact Evaluates.binary (av := number (s ^ 2)) (bv := number (c ^ 2))
    (Evaluates.binary (Evaluates.lookup getS) (Evaluates.lookup getS) (by simp [scalarBinary, number, pow_two]))
    (Evaluates.lookup getC2) (by simp [scalarBinary, number])

theorem root_evaluate (n : Nat) (env : ScalarEnv) (getD : env "d" = some (number n)) :
    Evaluates program env (.call (.name "isqrt") [.name "d"]) (number (Nat.sqrt n)) := by
  apply Evaluates.library_call (.cons (Evaluates.lookup getD) .nil) rfl
  change some (Math.isqrtCall [.int n]) = _
  have root : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  simp [Math.isqrtCall, root, number, show ¬(n : Int) < 0 by omega]

theorem square_evaluate (n : Nat) (env : ScalarEnv) (getD : env "d" = some (number n))
    (getT : env "t" = some (number (Nat.sqrt n))) :
    Evaluates program env squareCondition (.bool (decide (square n))) := by
  apply Evaluates.compare (av := number (Nat.sqrt n * Nat.sqrt n)) (bv := number n)
    (Evaluates.binary (Evaluates.lookup getT) (Evaluates.lookup getT) (by simp [scalarBinary, number]))
    (Evaluates.lookup getD)
  simp [scalarCompare, evalCmpOp, square]
  norm_cast

private theorem maximum_pair (a b : Int) :
    positionalMaximum [.int a, .int b] = some (.ok (.int (max a b))) := by
  simp [positionalMaximum, maximumFrom, scalarCompare, evalCmpOp, max_def]
  split <;> split <;> simp_all <;> omega

theorem lower_evaluate (c s : Nat) (env : ScalarEnv)
    (getS : env "s" = some (number s)) (getC : env "c" = some (number c)) :
    Evaluates program env lowerExpression (.int (max 1 ((s : Int) - c))) := by
  apply Evaluates.builtin (values := [.int 1, .int ((s : Int) - c)])
  · exact .cons Evaluates.constant
      (.cons (Evaluates.binary (Evaluates.lookup getS) (Evaluates.lookup getC) rfl) .nil)
  · exact maximum_pair _ _

theorem upper_evaluate (s : Nat) (env : ScalarEnv) (getS : env "s" = some (number s)) :
    Evaluates program env upperExpression (number (s / 2)) :=
  Evaluates.binary (Evaluates.lookup getS) Evaluates.constant (by simp [scalarBinary, number, Int.natCast_ediv])

theorem addition_evaluate (c s : Nat) (lower : 2 ≤ s) (upper : s ≤ 2 * c) (env : ScalarEnv)
    (getLo : env "lo" = some (.int (max 1 ((s : Int) - c))))
    (getHi : env "hi" = some (number (s / 2))) :
    Evaluates program env additionExpression (number (pairCount c s)) := by
  apply Evaluates.binary (av := .int (((s / 2 : Nat) : Int) - max 1 ((s : Int) - c))) (bv := .int 1)
    (Evaluates.binary (Evaluates.lookup getHi) (Evaluates.lookup getLo) rfl) Evaluates.constant
  simp [scalarBinary, number, pairCount_int_formula lower upper, Int.natCast_ediv]

end solvers.«86»
