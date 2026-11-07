import ProjectEulerStatements.Lib.DivisorBounds
import «.formalsw».solvers.«3.py.ast»
import data.«project-euler-statements».«.formalsw».data.py.project_euler_statements.«p3.py.ast»
import lang.py.FormalSW.CallsCostLoops
import Mathlib.Data.Nat.Sqrt
import Mathlib.NumberTheory.Divisors
import Mathlib.Tactic.Linarith

/-! Transition bounds for the freshly linked Euler 3 sources. Integer arithmetic
and the library `isqrt` operation are atomic; these are not bit-cost bounds. -/
namespace solvers.«3».complexity
open FormalSW.PyAst FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 1200000

private def weight (n : Nat) := min n 8 + Nat.sqrt n

private theorem weight_mono {a b : Nat} (h : a ≤ b) : weight a ≤ weight b := by
  have := Nat.sqrt_le_sqrt h
  have := min_le_min_right 8 h
  unfold weight
  omega

private theorem weight_div (n f : Nat) (positive : 0 < n) (factor : 2 ≤ f) :
    weight (n / f) < weight n := by
  have decreases := Nat.div_lt_self positive (by omega : 1 < f)
  have mono := Nat.sqrt_le_sqrt (Nat.div_le_self n f)
  by_cases small : n ≤ 8
  · simp only [weight, min_eq_left small, min_eq_left (by omega : n / f ≤ 8)]
    omega
  · have root : 3 ≤ Nat.sqrt n := Nat.le_sqrt.mpr (by omega)
    have upper := Nat.lt_succ_sqrt n
    have half : n / f ≤ n / 2 := Nat.div_le_div_left factor (by decide)
    have below : n / 2 < Nat.sqrt n * Nat.sqrt n := by
      apply (Nat.div_lt_iff_lt_mul (by decide : 0 < 2)).mpr
      nlinarith
    have smaller := Nat.sqrt_lt.mpr (lt_of_le_of_lt half below)
    have bound := min_le_left (n / f) 8
    have bound' := min_le_right (n / f) 8
    simp only [weight, min_eq_right (by omega : 8 ≤ n)]
    omega

private def modTest (factor : Expr) : Expr :=
  .compare (.binOp (.name "n") .mod factor) [(.eq, .constant (.int 0))]

private theorem mod_cost (program : Module) (n f : Nat) (positive : 0 < f)
    (env : ScalarEnv) (factor : Expr) (getN : env "n" = some (.int n))
    (getF : Cost.EvaluatesIn program env factor 1 (.int f)) :
    Cost.EvaluatesIn program env (modTest factor) 9 (.bool (decide (f ∣ n))) := by
  apply Cost.EvaluatesIn.compare (ca := 5) (cb := 1) (av := .int (n % f)) (bv := .int 0)
  · exact Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup getN) getF
      (by simp [scalarBinary, positive.ne'])
  · exact Cost.EvaluatesIn.constant
  · simp [scalarCompare, evalCmpOp, ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast]

namespace Optimized
private abbrev program := Solvers.P3Ast.module
private def stripBody (factor : Expr) : List Stmt :=
  [.assign [.name "largest"] factor, .augAssign (.name "n") .floorDiv factor]
private def finish : List Stmt :=
  [.ifThen (.compare (.name "n") [(.gt, .constant (.int 1))])
    [.assign [.name "largest"] (.name "n")] [], .return (.name "largest")]
private def outerTest : Expr :=
  .compare (.binOp (.name "f") .mult (.name "f")) [(.ltE, .name "n")]
private def outerBody : List Stmt :=
  [.whileDo (modTest (.name "f")) (stripBody (.name "f")) [],
   .augAssign (.name "f") .add (.constant (.int 2))]
private def outerLoop : List Stmt := .whileDo outerTest outerBody [] :: finish

/-- The factor expression is either a fixed literal or the unchanged local `f`. -/
private def Factor (factor : Expr) (f : Nat) (env : ScalarEnv) : Prop :=
  factor = .constant (.int f) ∨ factor = .name "f" ∧ env "f" = some (.int f)

private theorem Factor.cost (h : Factor factor f env) :
    Cost.EvaluatesIn program env factor 1 (.int f) := by
  rcases h with rfl | ⟨rfl, found⟩
  · exact Cost.EvaluatesIn.constant
  · exact Cost.EvaluatesIn.lookup found

private theorem Factor.preserve (h : Factor factor f env) (same : env' "f" = env "f") :
    Factor factor f env' := by
  rcases h with fixed | ⟨named, found⟩
  · exact Or.inl fixed
  · exact Or.inr ⟨named, same.trans found⟩

private theorem strip_cost (n f : Nat) (positive : 0 < n) (large : 2 ≤ f)
    (factor : Expr) (env : ScalarEnv) (getN : env "n" = some (.int n))
    (getF : Factor factor f env) (largest : Int) (getL : env "largest" = some (.int largest))
    (following : List Stmt) (stack : List Frame) :
    ∃ cost n' largest' env', 0 < n' ∧ n' ≤ n ∧
      env' "n" = some (.int n') ∧ env' "largest" = some (.int largest') ∧
      env' "f" = env "f" ∧ cost + 21 * weight n' ≤ 21 * weight n + 11 ∧
      ReachesIn (step program)
        ⟨.block (.whileDo (modTest factor) (stripBody factor) [] :: following) env, stack⟩ cost
        ⟨.block following env', stack⟩ := by
  induction n using Nat.strong_induction_on generalizing env largest with
  | h n ih =>
      have test := mod_cost program n f (by omega) env factor getN getF.cost
      by_cases divides : f ∣ n
      · let env' := (env.set "largest" (.int f)).set "n" (.int (n / f))
        have same : env' "f" = env "f" := by simp [env', ScalarEnv.set]
        have quotient : 0 < n / f := Nat.div_pos (Nat.le_of_dvd positive divides) (by omega)
        obtain ⟨cost, n', largest', final, pos, bound, en, el, ef, budget, run⟩ :=
          ih (n / f) (Nat.div_lt_self positive (by omega)) quotient env'
            (by simp [env', ScalarEnv.set]) (getF.preserve same) f
            (by simp [env', ScalarEnv.set])
        have first : ReachesIn (step program)
            ⟨.block (.whileDo (modTest factor) (stripBody factor) [] :: following) env, stack⟩ 21
            ⟨.block (.whileDo (modTest factor) (stripBody factor) [] :: following) env', stack⟩ := by
          have tested := Cost.whileTest (by simpa [divides] using test)
            (stripBody factor) [] following stack
          have assigned := Cost.assign getF.cost "largest"
            (.augAssign (.name "n") .floorDiv factor ::
              .whileDo (modTest factor) (stripBody factor) [] :: following) stack
          have divided := Cost.augAssign (name := "n") (op := .floorDiv) (expression := factor) (value := .int (n / f))
            (Cost.EvaluatesIn.binary
              (Cost.EvaluatesIn.lookup (by simpa [ScalarEnv.set] using getN))
              (getF.preserve (env' := env.set "largest" (.int f)) (by simp [ScalarEnv.set])).cost
              (by simp [scalarBinary, show f ≠ 0 by omega]))
            (.whileDo (modTest factor) (stripBody factor) [] :: following) stack
          exact (tested.trans assigned).trans divided
        refine ⟨21 + cost, n', largest', final, pos,
          le_trans bound (Nat.div_le_self _ _), en, el, ef.trans same, ?_, first.trans run⟩
        have := weight_div n f positive large
        omega
      · refine ⟨11, n, largest, env, positive, le_rfl, getN, getL, rfl, by omega, ?_⟩
        exact Cost.whileTest (by simpa [divides] using test) (stripBody factor) [] following stack

private theorem finish_cost (n : Nat) (largest : Int) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) (getL : env "largest" = some (.int largest))
    (stack : List Frame) :
    ∃ cost answer, cost ≤ 12 ∧ ReachesIn (step program) ⟨.block finish env, stack⟩ cost
      ⟨.value (.int answer), stack⟩ := by
  have test : Cost.EvaluatesIn program env (.compare (.name "n") [(.gt, .constant (.int 1))])
      5 (.bool (decide (1 < n))) :=
    Cost.EvaluatesIn.compare (Cost.EvaluatesIn.lookup getN) Cost.EvaluatesIn.constant
      (by simp [scalarCompare, evalCmpOp])
  by_cases more : 1 < n
  · refine ⟨12, n, by omega, ?_⟩
    have start := Cost.branch (by simpa [more] using test)
      [.assign [.name "largest"] (.name "n")] [] [.return (.name "largest")] stack
    have assign := Cost.assign (module := program) (Cost.EvaluatesIn.lookup getN) "largest" [.return (.name "largest")] stack
    exact (start.trans assign).trans (.next rfl (.next rfl .refl))
  · refine ⟨9, largest, by omega, ?_⟩
    have start := Cost.branch (by simpa [more] using test)
      [.assign [.name "largest"] (.name "n")] [] [.return (.name "largest")] stack
    exact start.trans (.next rfl (Cost.EvaluatesIn.lookup getL stack))

private theorem outer_cost (N n f : Nat) (positive : 0 < n) (bound : n ≤ N) (factor : 3 ≤ f)
    (largest : Int) (env : ScalarEnv) (getN : env "n" = some (.int n))
    (getF : env "f" = some (.int f)) (getL : env "largest" = some (.int largest))
    (stack : List Frame) :
    ∃ cost answer, cost ≤ 21 * weight n + 29 * (Nat.sqrt N + 1 - f) + 23 ∧
      ReachesIn (step program) ⟨.block outerLoop env, stack⟩ cost ⟨.value (.int answer), stack⟩ := by
  induction measure : Nat.sqrt N + 1 - f using Nat.strong_induction_on generalizing n f largest env with
  | h remaining ih =>
      have test : Cost.EvaluatesIn program env outerTest 9 (.bool (decide (f * f ≤ n))) := by
        apply Cost.EvaluatesIn.compare (ca := 5) (cb := 1) (av := .int (f * f))
        · exact Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup getF) (Cost.EvaluatesIn.lookup getF)
            (by simp [scalarBinary])
        · exact Cost.EvaluatesIn.lookup getN
        · simp only [scalarCompare, evalCmpOp, Option.some.injEq, decide_eq_decide]
          norm_cast
      by_cases active : f * f ≤ n
      · have within : f ≤ Nat.sqrt N := Nat.le_sqrt.mpr (le_trans active bound)
        let following := .augAssign (.name "f") .add (.constant (.int 2)) :: outerLoop
        obtain ⟨stripCost, n', largest', env', pos, smaller, en, el, ef, stripBound, strip⟩ :=
          strip_cost n f positive (by omega) (.name "f") env getN (Or.inr ⟨rfl, getF⟩)
            largest getL following stack
        let final := env'.set "f" (.int (f + 2 : Nat))
        obtain ⟨cost, answer, budget, run⟩ := ih (Nat.sqrt N + 1 - (f + 2)) (by omega)
          n' (f + 2) pos (le_trans smaller bound) (by omega) largest' final
          (by simpa [final, ScalarEnv.set] using en) (by simp [final, ScalarEnv.set])
          (by simpa [final, ScalarEnv.set] using el) rfl
        have tested := Cost.whileTest (by simpa [active] using test) outerBody [] finish stack
        have advance := Cost.augAssign (module := program) (name := "f") (op := .add) (expression := .constant (.int 2))
          (Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup (ef.trans getF)) Cost.EvaluatesIn.constant
            (by simp [scalarBinary] : scalarBinary .add (.int f) (.int 2) = some (.ok (.int (f + 2 : Nat)))))
          outerLoop stack
        refine ⟨11 + stripCost + 7 + cost, answer, ?_, ((tested.trans strip).trans advance).trans run⟩
        omega
      · obtain ⟨cost, answer, budget, run⟩ := finish_cost n largest env getN getL stack
        refine ⟨11 + cost, answer, by omega, ?_⟩
        exact (Cost.whileTest (test := false) (by simpa [active] using test) outerBody [] finish stack).trans run

end Optimized

/-- Includes both nested factor-stripping loops, without charging each division
as a fresh full scan. -/
theorem optimized_bound (N : Nat) (valid : 2 ≤ N) :
    ∃ cost value, cost ≤ 50 * Nat.sqrt N + 250 ∧
      FinishesIn (step Solvers.P3Ast.module) ⟨.invoke "solve" [.int N], []⟩ cost (.ok value) := by
  let env := (argumentEnv [⟨"n", some (.name "int"), none⟩] [.int N]).set "largest" (.int 1)
  let following := .assign [.name "f"] (.constant (.int 3)) :: Optimized.outerLoop
  obtain ⟨stripCost, n, largest, env', positive, smaller, en, el, _, stripBound, strip⟩ :=
    Optimized.strip_cost N 2 (by omega) (by decide) (.constant (.int 2)) env rfl (Or.inl rfl)
      1 rfl following []
  obtain ⟨cost, answer, budget, run⟩ := Optimized.outer_cost N n 3 positive smaller (by decide)
    largest (env'.set "f" (.int 3)) (by simpa [ScalarEnv.set] using en) rfl
    (by simpa [ScalarEnv.set] using el) []
  have lead : ReachesIn (step Solvers.P3Ast.module) ⟨.invoke "solve" [.int N], []⟩ 4
      ⟨.block (.whileDo (modTest (.constant (.int 2))) (Optimized.stripBody (.constant (.int 2))) [] :: following) env, []⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  have assign := Cost.assign (module := Solvers.P3Ast.module) (env := env')
    Cost.EvaluatesIn.constant "f" Optimized.outerLoop [] (value := .int 3)
  refine ⟨4 + stripCost + 3 + cost + 1, .int answer, ?_,
    (((lead.trans strip).trans assign).trans run).finishes (.done rfl)⟩
  have weightBound : weight N ≤ Nat.sqrt N + 8 := by unfold weight; omega
  omega

def optimized (n : Nat) : Execution Constant := function Solvers.P3Ast.module "solve" [.int n]
def reference (n : Nat) : Execution Constant := function P3Ast.module "naive" [.int n]

theorem optimized_bigO : Execution.IsBigO optimized (fun n => 2 ≤ n) id Nat.sqrt := by
  refine ⟨300, 2, fun n valid _ => ?_⟩
  obtain ⟨cost, value, bound, run⟩ := optimized_bound n valid
  have root : 1 ≤ Nat.sqrt n := Nat.le_sqrt.mpr (by omega)
  exact ⟨cost, .ok value, run, by dsimp; omega⟩

namespace Reference
private abbrev program := P3Ast.module
private def primeEnv (n : Nat) : ScalarEnv := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]
private def primeLoop : List Stmt :=
  (executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2).drop 5
private def primeBody : List Stmt :=
  [.ifThen (modTest (.name "p")) [.return (.constant (.bool false))] [],
   .augAssign (.name "p") .add (.constant (.int 2))]
private def primeTest : Expr := .compare (.name "p") [(.ltE, .name "r")]

private theorem prime_loop_cost (n r p : Nat) (positive : 0 < p) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) (getR : env "r" = some (.int r))
    (getP : env "p" = some (.int p)) (stack : List Frame) :
    ∃ cost answer, cost ≤ 25 * (r + 1 - p) + 20 ∧
      ReachesIn (step program) ⟨.block primeLoop env, stack⟩ cost ⟨.value (.bool answer), stack⟩ := by
  induction measure : r + 1 - p using Nat.strong_induction_on generalizing p env with
  | h remaining ih =>
      have test : Cost.EvaluatesIn program env primeTest 5 (.bool (decide (p ≤ r))) :=
        Cost.EvaluatesIn.compare (Cost.EvaluatesIn.lookup getP) (Cost.EvaluatesIn.lookup getR)
          (by simp [scalarCompare, evalCmpOp])
      by_cases active : p ≤ r
      · have tested := Cost.whileTest (test := true) (by simpa [active] using test)
          primeBody [] [.return (.constant (.bool true))] stack
        have remainder := mod_cost program n p positive env (.name "p") getN (Cost.EvaluatesIn.lookup getP)
        by_cases divides : p ∣ n
        · have branch := Cost.branch (test := true) (by simpa [divides] using remainder)
            [.return (.constant (.bool false))] []
            (.augAssign (.name "p") .add (.constant (.int 2)) :: primeLoop) stack
          refine ⟨20, false, by omega, ?_⟩
          exact (tested.trans branch).trans (.next rfl (.next rfl .refl))
        · have branch := Cost.branch (test := false) (by simpa [divides] using remainder)
            [.return (.constant (.bool false))] []
            (.augAssign (.name "p") .add (.constant (.int 2)) :: primeLoop) stack
          have advance := Cost.augAssign (module := program) (name := "p") (op := .add)
            (expression := .constant (.int 2))
            (Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup getP) Cost.EvaluatesIn.constant
              (by simp [scalarBinary] : scalarBinary .add (.int p) (.int 2) = some (.ok (.int (p + 2 : Nat)))))
            primeLoop stack
          obtain ⟨cost, answer, budget, run⟩ := ih (r + 1 - (p + 2)) (by omega)
            (p + 2) (by omega) (env.set "p" (.int (p + 2 : Nat)))
            (by simpa [ScalarEnv.set] using getN) (by simpa [ScalarEnv.set] using getR) rfl rfl
          exact ⟨25 + cost, answer, by omega, ((tested.trans branch).trans advance).trans run⟩
      · have tested := Cost.whileTest (test := false) (by simpa [active] using test)
          primeBody [] [.return (.constant (.bool true))] stack
        exact ⟨9, true, by omega, tested.trans (.next rfl (.next rfl .refl))⟩

private theorem prime_two (stack : List Frame) :
    ReachesIn (step program) ⟨.invoke "is_prime" [.int 2], stack⟩ 17
      ⟨.value (.bool true), stack⟩ := by
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_

private theorem prime_even (n : Nat) (large : 2 < n) (even : 2 ∣ n) (stack : List Frame) :
    ReachesIn (step program) ⟨.invoke "is_prime" [.int n], stack⟩ 28
      ⟨.value (.bool false), stack⟩ := by
  have small : ¬ (n : Int) < 2 := by omega
  have two : ¬ (n : Int) = 2 := by omega
  have parity : (n : Int) % 2 = 0 := by exact_mod_cast Nat.mod_eq_zero_of_dvd even
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  simp only [evalCmpOp, small, decide_false]
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  simp only [evalCmpOp, two, decide_false]
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  simp only [evalCmpOp, parity, decide_true]
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_

private theorem prime_odd_prefix (n : Nat) (large : 2 < n) (odd : ¬ 2 ∣ n) (stack : List Frame) :
    ReachesIn (step program) ⟨.invoke "is_prime" [.int n], stack⟩ 35
      ⟨.block primeLoop (((primeEnv n).set "r" (.int (Nat.sqrt n))).set "p" (.int 3)), stack⟩ := by
  have small : ¬ (n : Int) < 2 := by omega
  have two : ¬ (n : Int) = 2 := by omega
  have parity : (n : Int) % 2 ≠ 0 := by
    simpa [← Int.dvd_iff_emod_eq_zero, Int.ofNat_dvd_natCast] using odd
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  simp only [evalCmpOp, small, decide_false]
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  simp only [evalCmpOp, two, decide_false]
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  simp only [evalCmpOp, parity, decide_false]
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  have root : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  simp only [Int.toNat_natCast, root]
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_

private theorem prime_cost (n : Nat) (valid : 2 ≤ n) (stack : List Frame) :
    ∃ cost answer, cost ≤ 25 * Nat.sqrt n + 60 ∧
      ReachesIn (step program) ⟨.invoke "is_prime" [.int n], stack⟩ cost
        ⟨.value (.bool answer), stack⟩ := by
  by_cases two : n = 2
  · subst n
    exact ⟨17, true, by omega, prime_two stack⟩
  · by_cases even : 2 ∣ n
    · exact ⟨28, false, by omega, prime_even n (by omega) even stack⟩
    · obtain ⟨cost, answer, budget, run⟩ := prime_loop_cost n (Nat.sqrt n) 3 (by decide)
        (((primeEnv n).set "r" (.int (Nat.sqrt n))).set "p" (.int 3)) rfl rfl rfl stack
      exact ⟨35 + cost, answer, by omega, (prime_odd_prefix n (by omega) even stack).trans run⟩

private def scanTest : Expr := .boolOp .and
  [modTest (.name "k"), .call (.name "is_prime") [.name "k"]]
private def scanBody : List Stmt := [.ifThen scanTest [.return (.name "k")] []]
private def scanLoop (k : Nat) : List Stmt :=
  [.rangeStepLoop "k" k 1 (-1) scanBody [], .return (.constant (.int 0))]
private def scanFrames (k : Nat) (env : ScalarEnv) (stack : List Frame) : List Frame :=
  .boolean .and [.call (.name "is_prime") [.name "k"]] (env.set "k" (.int k)) ::
    .branch [.return (.name "k")] [] (scanLoop (k - 1)) (env.set "k" (.int k)) :: stack

private theorem scan_front (n k : Nat) (large : 2 ≤ k) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) (stack : List Frame) :
    ReachesIn (step program) ⟨.block (scanLoop k) env, stack⟩ 12
      ⟨.value (.bool (decide (k ∣ n))), scanFrames k env stack⟩ := by
  have first : ReachesIn (step program) ⟨.block (scanLoop k) env, stack⟩ 3
      ⟨.expr (modTest (.name "k")) (env.set "k" (.int k)), scanFrames k env stack⟩ := by
    refine .next (following := ⟨.block (scanBody ++ scanLoop (k - 1))
      (env.set "k" (.int k)), stack⟩) ?_ (.next rfl (.next rfl .refl))
    simp [step, scanLoop, scanBody, scanTest, rangeContinues,
      loopContinuation, loopControl, loopControlStmt, Int.natCast_sub (by omega : 1 ≤ k),
      show 1 < k by omega, Int.sub_eq_add_neg]
  exact first.trans (mod_cost program n k (by omega) _ (.name "k")
    (by simpa [ScalarEnv.set] using getN) (Cost.EvaluatesIn.lookup rfl) _)

private theorem scan_skip (n k : Nat) (large : 2 ≤ k) (absent : ¬ k ∣ n)
    (env : ScalarEnv) (getN : env "n" = some (.int n)) (stack : List Frame) :
    ReachesIn (step program) ⟨.block (scanLoop k) env, stack⟩ 14
      ⟨.block (scanLoop (k - 1)) (env.set "k" (.int k)), stack⟩ := by
  have front := scan_front n k large env getN stack
  simp only [absent, decide_false] at front
  exact front.trans (.next rfl (.next rfl .refl))

private def primeFrames (k : Nat) (env : ScalarEnv) (stack : List Frame) : List Frame :=
  .boolean .and [] (env.set "k" (.int k)) ::
    .branch [.return (.name "k")] [] (scanLoop (k - 1)) (env.set "k" (.int k)) :: stack

private theorem scan_call (n k : Nat) (large : 2 ≤ k) (divides : k ∣ n)
    (env : ScalarEnv) (getN : env "n" = some (.int n)) (stack : List Frame) :
    ReachesIn (step program) ⟨.block (scanLoop k) env, stack⟩ 16
      ⟨.invoke "is_prime" [.int k], primeFrames k env stack⟩ := by
  have front := scan_front n k large env getN stack
  simp only [divides, decide_true] at front
  exact front.trans (.next rfl (.next rfl (.next rfl (.next rfl .refl))))

private def divisorCount (n k : Nat) := ((Finset.range (k + 1)).filter (fun d => d ∣ n)).card

private theorem divisorCount_step (n k : Nat) :
    divisorCount n (k + 1) = divisorCount n k + if k + 1 ∣ n then 1 else 0 := by
  unfold divisorCount
  rw [show Finset.range (k + 1 + 1) = insert (k + 1) (Finset.range (k + 1)) from Finset.range_add_one, Finset.filter_insert]
  by_cases divides : k + 1 ∣ n
  all_goals simp [divides]

private theorem scan_cost (n k : Nat) (bound : k ≤ n) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) (stack : List Frame) :
    ∃ cost answer, cost ≤ 20 * k + (25 * Nat.sqrt n + 60) * divisorCount n k + 3 ∧
      ReachesIn (step program) ⟨.block (scanLoop k) env, stack⟩ cost
        ⟨.value (.int answer), stack⟩ := by
  induction k generalizing env with
  | zero =>
      refine ⟨3, 0, by omega, ?_⟩
      repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  | succ k ih =>
      by_cases large : 2 ≤ k + 1
      · obtain ⟨cost, answer, budget, run⟩ := ih (by omega) (env.set "k" (.int (k + 1 : Nat)))
          (by simpa [ScalarEnv.set] using getN)
        by_cases divides : k + 1 ∣ n
        · obtain ⟨primeCost, flag, primeBound, primeRun⟩ := prime_cost (k + 1) large
            (primeFrames (k + 1) env stack)
          have first := (scan_call n (k + 1) large divides env getN stack).trans primeRun
          have root := Nat.sqrt_le_sqrt bound
          have count := divisorCount_step n k
          simp only [divides, ↓reduceIte] at count
          cases flag with
          | false =>
              have tail : ReachesIn (step program)
                  ⟨.value (.bool false), primeFrames (k + 1) env stack⟩ 2
                  ⟨.block (scanLoop k) (env.set "k" (.int (k + 1 : Nat))), stack⟩ :=
                .next rfl (.next rfl .refl)
              refine ⟨16 + primeCost + 2 + cost, answer, ?_, (first.trans tail).trans run⟩
              rw [count, Nat.mul_add]
              nlinarith
          | true =>
              have tail : ReachesIn (step program)
                  ⟨.value (.bool true), primeFrames (k + 1) env stack⟩ 4
                  ⟨.value (.int (k + 1 : Nat)), stack⟩ :=
                .next rfl (.next rfl (.next rfl (.next rfl .refl)))
              refine ⟨16 + primeCost + 4, (k + 1 : Nat), ?_, first.trans tail⟩
              rw [count, Nat.mul_add]
              nlinarith
        · have count := divisorCount_step n k
          simp only [divides, ↓reduceIte, Nat.add_zero] at count
          refine ⟨14 + cost, answer, ?_, (scan_skip n (k + 1) large divides env getN stack).trans run⟩
          rw [count]
          omega
      · have eq : k = 0 := by omega
        subst k
        refine ⟨3, 0, by omega, ?_⟩
        repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_

private theorem scan_lead (n : Nat) :
    ReachesIn (step program) ⟨.invoke "naive" [.int n], []⟩ 11
      ⟨.block (scanLoop n) (argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]), []⟩ := by
  repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_

private theorem high_scan (m remaining : Nat) (valid : 2 ≤ m) (short : remaining < m)
    (env : ScalarEnv) (getN : env "n" = some (.int (2 * m : Nat))) :
    ∃ final, ReachesIn (step program) ⟨.block (scanLoop (m + remaining)) env, []⟩ (14 * remaining)
      ⟨.block (scanLoop m) final, []⟩ := by
  induction remaining generalizing env with
  | zero => exact ⟨env, .refl⟩
  | succ remaining ih =>
      have absent : ¬ m + (remaining + 1) ∣ 2 * m := by
        intro divides
        have same := Nat.eq_of_dvd_of_lt_two_mul (by omega : 2 * m ≠ 0) divides (by omega)
        omega
      obtain ⟨final, run⟩ := ih (by omega) (env.set "k" (.int (m + (remaining + 1) : Nat)))
        (by simpa [ScalarEnv.set] using getN)
      have first := scan_skip (2 * m) (m + (remaining + 1)) (by omega) absent env getN []
      refine ⟨final, ?_⟩
      have index : m + (remaining + 1) - 1 = m + remaining := by omega
      rw [index] at first
      simpa [Nat.mul_add, Nat.add_comm] using first.trans run

/-- An even input is rejected first, then every candidate strictly above half
of that input fails divisibility. This is a prefix of the actual reference run. -/
private theorem even_prefix (m : Nat) (valid : 2 ≤ m) :
    ∃ final, ReachesIn (step program) ⟨.invoke "naive" [.int (2 * m : Nat)], []⟩ (14 * m + 43)
      ⟨.block (scanLoop m) final, []⟩ := by
  let env := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int (2 * m : Nat)]
  have call := scan_call (2 * m) (2 * m) (by omega) dvd_rfl env rfl []
  have prime := prime_even (2 * m) (by omega) (by omega) (primeFrames (2 * m) env [])
  have tail : ReachesIn (step program) ⟨.value (.bool false), primeFrames (2 * m) env []⟩ 2
      ⟨.block (scanLoop (2 * m - 1)) (env.set "k" (.int (2 * m : Nat))), []⟩ :=
    .next rfl (.next rfl .refl)
  obtain ⟨final, rest⟩ := high_scan m (m - 1) valid (by omega)
    (env.set "k" (.int (2 * m : Nat))) rfl
  have index : m + (m - 1) = 2 * m - 1 := by omega
  rw [index] at rest
  refine ⟨final, ?_⟩
  have cost : 14 * m + 43 = ((11 + 16) + 28) + 2 + 14 * (m - 1) := by omega
  rw [cost]
  exact ((((scan_lead (2 * m)).trans call).trans prime).trans tail).trans rest

end Reference

/-- Only divisors invoke primality testing; the total helper budget is linear. -/
theorem reference_bound (n : Nat) (valid : 2 ≤ n) :
    ∃ cost value, cost ≤ 190 * n + 15 ∧
      FinishesIn (step P3Ast.module) ⟨.invoke "naive" [.int n], []⟩ cost (.ok value) := by
  obtain ⟨cost, answer, budget, run⟩ := Reference.scan_cost n n le_rfl
    (argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]) rfl []
  have count : Reference.divisorCount n n ≤ 2 * Nat.sqrt n := by
    simpa only [Reference.divisorCount, Nat.filter_dvd_eq_divisors (by omega : n ≠ 0)] using
      ProjectEulerStatements.DivisorBounds.card_le_twice_sqrt n (by omega)
  have helpers := Nat.mul_le_mul_left (25 * Nat.sqrt n + 60) count
  have square := Nat.sqrt_le n
  have root := Nat.sqrt_le_self n
  refine ⟨11 + cost + 1, .int answer, ?_, ((Reference.scan_lead n).trans run).finishes (.done rfl)⟩
  nlinarith

theorem reference_bigO : Execution.IsBigO reference (fun n => 2 ≤ n) id id := by
  refine ⟨200, 2, fun n valid _ => ?_⟩
  obtain ⟨cost, value, bound, run⟩ := reference_bound n valid
  exact ⟨cost, .ok value, run, by dsimp; omega⟩

private theorem prefix_lt {transition : σ → Step σ α} {start final : σ}
    (path : ReachesIn transition start prefixCost final)
    (run : FinishesIn transition start cost result) : prefixCost < cost := by
  induction path generalizing cost result with
  | refl => exact run.positive
  | next next _ ih =>
      cases run with
      | done done => simp [next] at done
      | next next' tail =>
          simp only [next, Step.next.injEq] at next'
          subst next'
          have := ih tail
          omega

/-- A linear worst-case lower bound, witnessed by every even input at least four. -/
theorem reference_even_lower (m : Nat) (valid : 2 ≤ m)
    (run : FinishesIn (step P3Ast.module) ⟨.invoke "naive" [.int (2 * m : Nat)], []⟩ cost result) :
    14 * m + 43 < cost := by
  obtain ⟨_, path⟩ := Reference.even_prefix m valid
  exact prefix_lt path run

/-- The reference cannot satisfy the optimized solver's square-root upper bound. -/
theorem reference_not_sqrt :
    ¬ Execution.IsBigO reference (fun n => 2 ≤ n) id Nat.sqrt := by
  rintro ⟨constant, threshold, bounded⟩
  let t := constant + threshold + 2
  let m := t * t
  have tLarge : 2 ≤ t := by dsimp [t]; omega
  have mLarge : 2 ≤ m := by dsimp [m]; nlinarith
  have inputLarge : threshold ≤ 2 * m := by dsimp [m, t]; nlinarith
  obtain ⟨cost, outcome, run, small⟩ := bounded (2 * m) (by omega) inputLarge
  have lower := reference_even_lower m mLarge run
  have root : Nat.sqrt (2 * m) ≤ 2 * t := by
    have top := Nat.sqrt_le (2 * m)
    dsimp [m] at top
    nlinarith
  have upper := Nat.mul_le_mul_left constant root
  change cost ≤ constant * Nat.sqrt (2 * m) at small
  dsimp [m, t] at lower upper
  nlinarith

/-- In particular, a constant reference bound is impossible. -/
theorem reference_not_constant :
    ¬ Execution.IsBigO reference (fun n => 2 ≤ n) id (fun _ => 1) := by
  rintro ⟨constant, threshold, bounded⟩
  apply reference_not_sqrt
  refine ⟨constant, max threshold 2, fun n valid large => ?_⟩
  have run := bounded n valid (by dsimp at large ⊢; omega)
  have root : 1 ≤ Nat.sqrt n := Nat.le_sqrt.mpr (by omega)
  apply run.mono
  dsimp
  simpa using Nat.mul_le_mul_left constant root

end solvers.«3».complexity
