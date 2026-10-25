import «.formalsw».solvers.«45.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p45.py»
import ProjectEulerSolutions.Lib.TriPentHexSearch

set_option Elab.async false

namespace solvers.«45»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.TriPentHex ProjectEulerSolutions.TriPentHexSearch
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
private def body : List Stmt :=
  [.assign [.name "h"] (.call (.name "hexagonal") [.name "n"]),
   .ifThen (.call (.name "is_pentagonal") [.name "h"]) [.return (.name "h")] [],
   .augAssign (.name "n") .add (.constant (.int 1))]
private def searchLoop : Stmt := .whileDo (.constant (.bool true)) body []

def sourceBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 1 with
  | .whileDo _ body _ :: _ => body
  | _ => []

theorem body_source : body=sourceBody := rfl

/-- A decreasing distance to the certified witness proves termination of the actual unbounded while loop. -/
theorem loop (next count : Nat) (equation : next+count=winner) (lower : firstHex≤next)
    (env : ScalarEnv) (getN : env "n"=some (natural next)) :
    BlockReturns program env [searchLoop] (natural (hexagonal winner)) := by
  induction count generalizing next env with
  | zero =>
      have equal : next=winner := by omega
      subst next
      let ready := env.set "h" (natural (hexagonal winner))
      have assigned : Evaluates program env (.call (.name "hexagonal") [.name "n"]) (natural (hexagonal winner)) :=
        Evaluates.call (.cons (Evaluates.lookup getN) .nil) (by simpa only [hex_nat] using hexagonal.result (winner : Int))
      have tested : Evaluates program ready (.call (.name "is_pentagonal") [.name "h"]) (flag (hexTest winner)) :=
        Evaluates.call (values := [natural (hexagonal winner)]) (.cons (Evaluates.lookup (by simp [ready, ScalarEnv.set])) .nil)
          (is_pentagonal.result (hexagonal winner))
      apply BlockReturns.while_loop (value := .bool true) Evaluates.constant
      apply BlockReturns.assign assigned
      apply BlockReturns.branch tested
      simp only [winner_accepted, scalarTruth, if_true, List.singleton_append]
      exact BlockReturns.return_value (Evaluates.lookup (by simp [ready, ScalarEnv.set]))
  | succ count ih =>
      let ready := env.set "h" (natural (hexagonal next))
      have assigned : Evaluates program env (.call (.name "hexagonal") [.name "n"]) (natural (hexagonal next)) :=
        Evaluates.call (.cons (Evaluates.lookup getN) .nil) (by simpa only [hex_nat] using hexagonal.result (next : Int))
      have tested : Evaluates program ready (.call (.name "is_pentagonal") [.name "h"]) (flag (hexTest next)) :=
        Evaluates.call (values := [natural (hexagonal next)]) (.cons (Evaluates.lookup (by simp [ready, ScalarEnv.set])) .nil)
          (is_pentagonal.result (hexagonal next))
      have rejected := no_earlier next lower (by omega)
      apply BlockReturns.while_loop (value := .bool true) Evaluates.constant
      apply BlockReturns.assign assigned
      apply BlockReturns.branch tested
      simp only [rejected, scalarTruth, Bool.false_eq_true, if_false, List.nil_append]
      apply BlockReturns.augAssign (value := natural (next+1))
        (Evaluates.binary (Evaluates.lookup (by simpa [ready, ScalarEnv.set] using getN)) Evaluates.constant (by simp [scalarBinary]))
      have rest := ih (next+1) (by omega) (by omega) (ready.set "n" (natural (next+1))) (by simp [ScalarEnv.set])
      simpa only [loopContinuation, show loopControl body=false from rfl, Bool.false_eq_true, if_false] using rest

/-- The optimized Python source returns the certified least common polygonal value. -/
theorem result : ReturnsValue program "solve" [] (natural (hexagonal winner)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  exact loop firstHex (winner-firstHex) rfl (by omega) _ rfl
/-- The returned value is least among all common triangular values beyond the problem's given example. -/
theorem least (index : Nat) (later : 285<index) (accepted : common index=true) :
    hexagonal winner ≤ triangular index := by
  have bound : triangleIndex≤index := by
    by_contra earlier
    have rejected := common_rejected index (by omega) (by omega)
    rw [rejected] at accepted
    cases accepted
  have lower := triangular_strictMono.monotone bound
  have same : triangular triangleIndex=hexagonal winner := hex_triangular winner (by decide)
  simpa only [same] using lower
end solve

theorem solve.«Equiv(naive_default)» :
    (Calls.function program "solve" []).Equiv (HeapCalls.scalarFunction P45Ast.module "naive_default" []) := by
  have optimized := solve.result.finishes
  have reference := HeapCalls.ReturnsValue.scalar_finishes (values := [])
    (data.«project-euler-statements».data.py.project_euler_statements.p45.naive_default.search_result [])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem solve.«Supported» :
    (Calls.function program "solve" []).Supported ∧ (HeapCalls.scalarFunction P45Ast.module "naive_default" []).Supported :=
  ⟨solve.result.returns.supported, data.«project-euler-statements».data.py.project_euler_statements.p45.naive_default.«Supported»⟩

end solvers.«45»
