import «.formalsw».solvers.«45.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p45.py»
import ProjectEulerSolutions.Lib.TriPentHexUnbounded

set_option Elab.async false

namespace solvers.«45»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.TriPentHex ProjectEulerSolutions.TriPentHexUnbounded
open data.«project-euler-statements».data.py.project_euler_statements.p45 (hexPolynomial hex_nat)
private abbrev program := Solvers.P45Ast.module
private abbrev number (n : Int) : Constant := .int n
private abbrev natural (n : Nat) : Constant := number n
private abbrev flag (b : Bool) : Constant := .bool b

namespace hexagonal
/-- The optimized source evaluates the exact arbitrary-precision hexagonal polynomial. -/
theorem result (n : Int) : ReturnsValue program "hexagonal" [number n] (number (hexPolynomial n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.binary (av := number n) (bv := number (2*n-1)) (Evaluates.lookup rfl)
  · exact Evaluates.binary (av := number (2*n)) (bv := number 1)
      (Evaluates.binary (av := number 2) (bv := number n) Evaluates.constant (Evaluates.lookup rfl) rfl) Evaluates.constant rfl
  · rfl
end hexagonal

namespace is_pentagonal
private def nonsquare : Expr := .compare (.binOp (.name "s") .mult (.name "s")) [(.notEq, .name "d")]
private def remainder : Expr := .compare
  (.binOp (.binOp (.constant (.int 1)) .add (.name "s")) .mod (.constant (.int 6))) [(.eq, .constant (.int 0))]

/-- The source's isqrt, nonsquare early return and congruence test implement the pentagonal criterion. -/
theorem result (x : Nat) :
    ReturnsValue program "is_pentagonal" [natural x] (flag (ProjectEulerSolutions.PentagonalDiscriminant.discriminant x)) := by
  let initial := argumentEnv ((findFunctionIn program.body "is_pentagonal").getD ([], [])).1 [natural x]
  let d := 1+24*x
  let root := Nat.sqrt d
  let discriminated := initial.set "d" (natural d)
  let rooted := discriminated.set "s" (natural root)
  have setD : Evaluates program initial
      (.binOp (.constant (.int 1)) .add (.binOp (.constant (.int 24)) .mult (.name "x"))) (natural d) :=
    Evaluates.binary (av := natural 1) (bv := natural (24*x)) Evaluates.constant
      (Evaluates.binary Evaluates.constant (Evaluates.lookup rfl) (by simp [scalarBinary])) (by simp [scalarBinary, d])
  have getD : discriminated "d"=some (natural d) := by simp [discriminated, ScalarEnv.set]
  have sqrtResult : Evaluates program discriminated (.call (.attribute (.name "math") "isqrt") [.name "d"]) (natural root) := by
    apply Evaluates.qualified_call
    apply Evaluates.library_call (.cons (Evaluates.lookup getD) .nil) rfl
    change some (Math.isqrtCall [.int d]) = _
    have same : Math.isqrt d=Nat.sqrt d := Nat.eq_sqrt.mpr (Math.isqrt_spec d)
    simp [Math.isqrtCall, same, root, show ¬ (d : Int)<0 by omega]
  have tested : Evaluates program rooted nonsquare (flag (decide (root*root≠d))) := by
    apply Evaluates.compare (av := natural (root*root)) (bv := natural d)
      (Evaluates.binary (av := natural root) (bv := natural root)
        (Evaluates.lookup (by simp [rooted, ScalarEnv.set])) (Evaluates.lookup (by simp [rooted, ScalarEnv.set]))
        (by simp [scalarBinary]))
      (Evaluates.lookup (by simp [rooted, discriminated, ScalarEnv.set]))
    change some (decide (((root*root : Nat) : Int)≠(d : Int))) = _
    simp only [ne_eq, Int.natCast_inj]
  have finalTest : Evaluates program rooted remainder (flag (decide ((1+root)%6=0))) := by
    apply Evaluates.compare (av := natural ((1+root)%6)) (bv := natural 0)
    · exact Evaluates.binary (av := natural (1+root)) (bv := natural 6)
        (Evaluates.binary (av := natural 1) (bv := natural root) Evaluates.constant
          (Evaluates.lookup (by simp [rooted, ScalarEnv.set])) (by simp [scalarBinary]))
        Evaluates.constant (by simp [scalarBinary])
    · exact Evaluates.constant
    · change some (decide ((((1+root)%6 : Nat) : Int)=0)) = _
      simp only [Int.natCast_eq_zero]
  have same : ProjectEulerSolutions.PentagonalDiscriminant.discriminant x=decide (root*root=d ∧ (1+root)%6=0) := by
    by_cases zero : x=0
    · subst x
      decide +kernel
    · simp [ProjectEulerSolutions.PentagonalDiscriminant.discriminant, zero, root, d, Nat.add_comm]
  rw [same]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign setD
  apply BlockReturns.assign sqrtResult
  by_cases square : root*root=d
  · simp only [square, true_and]
    apply BlockReturns.branch tested
    simpa only [flag, scalarTruth, square, ne_eq, not_true_eq_false, decide_false, Bool.false_eq_true,
      if_false, List.nil_append] using BlockReturns.return_value finalTest
  · simp only [square, false_and, decide_false]
    apply BlockReturns.branch tested
    simp only [flag, scalarTruth, square, ne_eq, not_false_eq_true, decide_true, if_true, List.singleton_append]
    exact BlockReturns.return_value Evaluates.constant
end is_pentagonal

namespace solve
private def predicate : Expr := .boolOp .and
  [.compare (.name "value") [(.gt, .name "after")], .call (.name "is_pentagonal") [.name "value"]]
private def body : List Stmt :=
  [.assign [.name "value"] (.call (.name "hexagonal") [.name "n"]),
    .ifThen predicate [.return (.name "value")] [],
    .augAssign (.name "n") .add (.constant (.int 1))]
private def searchLoop : Stmt := .whileDo (.constant (.bool true)) body []

private theorem predicate_result (after : Int) (n : Nat) (env : ScalarEnv)
    (getAfter : env "after" = some (number after)) (getValue : env "value" = some (natural (hexagonal n))) :
    Evaluates program env predicate (flag (decide (after < (hexagonal n : Int) ∧ hexTest n = true))) := by
  have first : Evaluates program env (.compare (.name "value") [(.gt, .name "after")])
      (flag (decide (after < (hexagonal n : Int)))) :=
    Evaluates.compare (Evaluates.lookup getValue) (Evaluates.lookup getAfter) (by simp [scalarCompare, evalCmpOp])
  have second : Evaluates program env (.call (.name "is_pentagonal") [.name "value"]) (flag (hexTest n)) :=
    Evaluates.call (.cons (Evaluates.lookup getValue) .nil) (is_pentagonal.result (hexagonal n))
  by_cases later : after < (hexagonal n : Int)
  · apply Evaluates.boolean_continue (by simpa only [later, decide_true] using first) rfl
    apply Evaluates.boolean_last
    simpa [later] using second
  · simp only [later, false_and, decide_false]
    exact Evaluates.boolean_stop (by simpa only [later, decide_false] using first) rfl

/-- The search reaches the least accepted index, whose existence follows from the recurrence. -/
theorem loop (after : Int) (n : Nat) (positive : 0 < n) (bounded : n ≤ winner after)
    (env : ScalarEnv) (getAfter : env "after" = some (number after)) (getN : env "n" = some (natural n)) :
    BlockReturns program env [searchLoop] (natural (answer after)) := by
  let ready := env.set "value" (natural (hexagonal n))
  have assigned : Evaluates program env (.call (.name "hexagonal") [.name "n"]) (natural (hexagonal n)) :=
    Evaluates.call (.cons (Evaluates.lookup getN) .nil) (by simpa only [hex_nat] using hexagonal.result (n : Int))
  have tested := predicate_result after n ready (by simpa [ready, ScalarEnv.set] using getAfter)
    (by simp [ready, ScalarEnv.set])
  by_cases selected : after < (hexagonal n : Int) ∧ hexTest n = true
  · rw [decide_eq_true selected] at tested
    have least := winner_minimal after n ⟨positive, selected⟩
    have same : n = winner after := by omega
    apply BlockReturns.while_loop (value := .bool true) Evaluates.constant
    apply BlockReturns.assign assigned
    apply BlockReturns.branch tested
    apply BlockReturns.return_value
    apply Evaluates.lookup
    simp [ready, ScalarEnv.set, answer, same]
  · rw [decide_eq_false selected] at tested
    have before : n < winner after := by
      by_contra late
      have same : n = winner after := by omega
      exact selected (by simpa only [same] using (winner_spec after).2)
    have rest := loop after (n + 1) (by omega) (by omega) (ready.set "n" (natural (n + 1)))
      (by simpa [ready, ScalarEnv.set] using getAfter) (by simp [ScalarEnv.set])
    apply BlockReturns.while_loop (value := .bool true) Evaluates.constant
    apply BlockReturns.assign assigned
    apply BlockReturns.branch tested
    apply BlockReturns.augAssign (value := natural (n + 1))
      (Evaluates.binary (Evaluates.lookup (by simpa [ready, ScalarEnv.set] using getN)) Evaluates.constant
        (by simp [scalarBinary]))
    simpa only [loopContinuation, show loopControl body = false from rfl, Bool.false_eq_true, if_false] using rest
termination_by winner after - n
decreasing_by omega

/-- The complete optimized source terminates for every signed threshold. -/
theorem result (after : Int) : ReturnsValue program "solve" [number after] (natural (answer after)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  exact loop after 1 (by decide) (winner_spec after).1 _ rfl rfl

theorem «Returns» (after : Int) : (Calls.function program "solve" [number after]).Returns := (result after).returns

end solve
end solvers.«45»

namespace solvers.«45».solve
open FormalSW.PyAst
open ProjectEulerSolutions.TriPentHexUnbounded

theorem «Equiv(naive)» (after : Int) :
    (Calls.function Solvers.P45Ast.module "solve" [.int after]).Equiv
      (Calls.function P45Ast.module "naive" [.int after]) := by
  have optimized := (result after).finishes
  have reference := (data.«project-euler-statements».data.py.project_euler_statements.p45.naive.result after).finishes
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ after : Int, (Calls.function Solvers.P45Ast.module "solve" [.int after]).Supported) ∧
      (∀ after : Int, (Calls.function P45Ast.module "naive" [.int after]).Supported) :=
  ⟨fun after => («Returns» after).supported,
    fun after => data.«project-euler-statements».data.py.project_euler_statements.p45.naive.«Supported» after⟩

end solvers.«45».solve
