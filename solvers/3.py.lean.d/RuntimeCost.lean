import «.formalsw».solvers.«3.py.ast»
import lang.py.FormalSW.CallsRuntimeCostBitBounds
import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Linarith

/-! Bit-sensitive runtime bound for trial division on the generated Python AST. -/
namespace solvers.«3».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.Calls
open Runtime.Scaled Runtime.BitBounds
set_option maxRecDepth 4096
set_option maxHeartbeats 1200000

private def weight (n : Nat) := min n 8 + Nat.sqrt n

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

private theorem Factor.cost (N : Nat) (h : Factor factor f env) :
    EvaluatesIn .bits program (unit N) env factor 1 (.int f) := by
  rcases h with rfl | ⟨rfl, found⟩
  · exact (EvaluatesIn.constant (unit_positive N))
  · exact EvaluatesIn.lookup (unit_positive N) found

private theorem Factor.preserve (h : Factor factor f env) (same : env' "f" = env "f") :
    Factor factor f env' := by
  rcases h with fixed | ⟨named, found⟩
  · exact Or.inl fixed
  · exact Or.inr ⟨named, same.trans found⟩

private theorem strip_cost (N n f : Nat) (positive : 0 < n) (large : 2 ≤ f)
    (nb : n ≤ N) (fb : f ≤ N+2)
    (factor : Expr) (env : ScalarEnv) (getN : env "n" = some (.int n))
    (getF : Factor factor f env) (largest : Int) (getL : env "largest" = some (.int largest))
    (following : List Stmt) (stack : List Frame) :
    ∃ cost n' largest' env', 0 < n' ∧ n' ≤ n ∧
      env' "n" = some (.int n') ∧ env' "largest" = some (.int largest') ∧
      env' "f" = env "f" ∧ cost + 21 * weight n' ≤ 21 * weight n + 11 ∧
      Reaches .bits program (unit N)
        ⟨.block (.whileDo (modTest factor) (stripBody factor) [] :: following) env, stack⟩ cost
        ⟨.block following env', stack⟩ := by
  induction n using Nat.strong_induction_on generalizing env largest with
  | h n ih =>
      have test := mod_cost program N n f (by omega) nb fb env factor getN (getF.cost N)
      by_cases divides : f ∣ n
      · let env' := (env.set "largest" (.int f)).set "n" (.int (n / f))
        have same : env' "f" = env "f" := by simp [env', ScalarEnv.set]
        have quotient : 0 < n / f := Nat.div_pos (Nat.le_of_dvd positive divides) (by omega)
        obtain ⟨cost, n', largest', final, pos, bound, en, el, ef, budget, run⟩ :=
          ih (n / f) (Nat.div_lt_self positive (by omega)) quotient (le_trans (Nat.div_le_self _ _) nb) env'
            (by simp [env', ScalarEnv.set]) (getF.preserve same) f
            (by simp [env', ScalarEnv.set])
        have first : Reaches .bits program (unit N)
            ⟨.block (.whileDo (modTest factor) (stripBody factor) [] :: following) env, stack⟩ 21
            ⟨.block (.whileDo (modTest factor) (stripBody factor) [] :: following) env', stack⟩ := by
          have tested := Runtime.Scaled.whileTest (unit_positive N) (by simpa [divides] using test)
            (stripBody factor) [] following stack
          have assigned := Runtime.Scaled.assign (unit_positive N) (getF.cost N) "largest"
            (.augAssign (.name "n") .floorDiv factor ::
              .whileDo (modTest factor) (stripBody factor) [] :: following) stack
          have divided := Runtime.Scaled.augAssign (unit_positive N) (name := "n") (op := .floorDiv) (expression := factor) (value := .int (n / f))
            (EvaluatesIn.binary (unit_positive N)
              (EvaluatesIn.lookup (unit_positive N) (by simpa [ScalarEnv.set] using getN))
              ((getF.preserve (env' := env.set "largest" (.int f)) (by simp [ScalarEnv.set])).cost N)
              (by simp [scalarBinary, show f ≠ 0 by omega])
              (binary_price N n f (cap N n (by omega)) (cap N f fb) .floorDiv (by simp)))
            (.whileDo (modTest factor) (stripBody factor) [] :: following) stack
          exact (tested.trans assigned).trans divided
        refine ⟨21 + cost, n', largest', final, pos,
          le_trans bound (Nat.div_le_self _ _), en, el, ef.trans same, ?_, first.trans run⟩
        have := weight_div n f positive large
        omega
      · refine ⟨11, n, largest, env, positive, le_rfl, getN, getL, rfl, by omega, ?_⟩
        exact Runtime.Scaled.whileTest (unit_positive N) (by simpa [divides] using test) (stripBody factor) [] following stack

private theorem finish_cost (N n : Nat) (nb : n ≤ N) (largest : Int) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) (getL : env "largest" = some (.int largest))
    (stack : List Frame) :
    ∃ cost answer, cost ≤ 12 ∧ Reaches .bits program (unit N) ⟨.block finish env, stack⟩ cost
      ⟨.value (.int answer), stack⟩ := by
  have test : EvaluatesIn .bits program (unit N) env (.compare (.name "n") [(.gt, .constant (.int 1))])
      5 (.bool (decide (1 < n))) :=
    EvaluatesIn.compare (unit_positive N) (EvaluatesIn.lookup (unit_positive N) getN) (EvaluatesIn.constant (unit_positive N))
      (by simp [scalarCompare, evalCmpOp])
      (compare_price N n 1 (cap N n (by omega)) (cap N 1 (by omega)))
  by_cases more : 1 < n
  · refine ⟨12, n, by omega, ?_⟩
    have start := Runtime.Scaled.branch (unit_positive N) (by simpa [more] using test)
      [.assign [.name "largest"] (.name "n")] [] [.return (.name "largest")] stack
    have assign := Runtime.Scaled.assign (unit_positive N) (arithmetic := .bits) (module := program) (EvaluatesIn.lookup (unit_positive N) getN) "largest" [.return (.name "largest")] stack
    exact (start.trans assign).trans (next_one rfl rfl (next_one rfl rfl .refl))
  · refine ⟨9, largest, by omega, ?_⟩
    have start := Runtime.Scaled.branch (unit_positive N) (by simpa [more] using test)
      [.assign [.name "largest"] (.name "n")] [] [.return (.name "largest")] stack
    exact start.trans (next_one rfl rfl ((EvaluatesIn.lookup (unit_positive N) getL).reaches stack))

private theorem outer_cost (N n f : Nat) (positive : 0 < n) (bound : n ≤ N) (factor : 3 ≤ f) (fb : f ≤ N+2)
    (largest : Int) (env : ScalarEnv) (getN : env "n" = some (.int n))
    (getF : env "f" = some (.int f)) (getL : env "largest" = some (.int largest))
    (stack : List Frame) :
    ∃ cost answer, cost ≤ 21 * weight n + 29 * (Nat.sqrt N + 1 - f) + 23 ∧
      Reaches .bits program (unit N) ⟨.block outerLoop env, stack⟩ cost ⟨.value (.int answer), stack⟩ := by
  induction measure : Nat.sqrt N + 1 - f using Nat.strong_induction_on generalizing n f largest env with
  | h remaining ih =>
      have test : EvaluatesIn .bits program (unit N) env outerTest 9 (.bool (decide (f * f ≤ n))) := by
        apply EvaluatesIn.compare (unit_positive N) (ca := 5) (cb := 1) (av := .int (f * f))
        · exact EvaluatesIn.binary (unit_positive N) (EvaluatesIn.lookup (unit_positive N) getF) (EvaluatesIn.lookup (unit_positive N) getF)
            (by simp [scalarBinary])
            (binary_price N f f (cap N f fb) (cap N f fb) .mult (by simp))
        · exact EvaluatesIn.lookup (unit_positive N) getN
        · simp only [scalarCompare, evalCmpOp, Option.some.injEq, decide_eq_decide]
          norm_cast
        · exact compare_price N (f*f) n (by nlinarith) (cap N n (by omega))
      by_cases active : f * f ≤ n
      · have within : f ≤ Nat.sqrt N := Nat.le_sqrt.mpr (le_trans active bound)
        let following := .augAssign (.name "f") .add (.constant (.int 2)) :: outerLoop
        obtain ⟨stripCost, n', largest', env', pos, smaller, en, el, ef, stripBound, strip⟩ :=
          strip_cost N n f positive (by omega) bound fb (.name "f") env getN (Or.inr ⟨rfl, getF⟩)
            largest getL following stack
        let final := env'.set "f" (.int (f + 2 : Nat))
        obtain ⟨cost, answer, budget, run⟩ := ih (Nat.sqrt N + 1 - (f + 2)) (by omega)
          n' (f + 2) pos (le_trans smaller bound) (by omega) (by have := Nat.sqrt_le_self N; omega) largest' final
          (by simpa [final, ScalarEnv.set] using en) (by simp [final, ScalarEnv.set])
          (by simpa [final, ScalarEnv.set] using el) rfl
        have tested := Runtime.Scaled.whileTest (unit_positive N) (by simpa [active] using test) outerBody [] finish stack
        have advance := Runtime.Scaled.augAssign (unit_positive N) (module := program) (name := "f") (op := .add) (expression := .constant (.int 2))
          (EvaluatesIn.binary (unit_positive N) (EvaluatesIn.lookup (unit_positive N) (ef.trans getF)) (EvaluatesIn.constant (unit_positive N))
            (by simp [scalarBinary] : scalarBinary .add (.int f) (.int 2) = some (.ok (.int (f + 2 : Nat))))
            (binary_price N f 2 (cap N f fb) (cap N 2 (by omega)) .add (by simp)))
          outerLoop stack
        refine ⟨11 + stripCost + 7 + cost, answer, ?_, ((tested.trans strip).trans advance).trans run⟩
        omega
      · obtain ⟨cost, answer, budget, run⟩ := finish_cost N n bound largest env getN getL stack
        refine ⟨11 + cost, answer, by omega, ?_⟩
        exact (Runtime.Scaled.whileTest (unit_positive N) (test := false) (by simpa [active] using test) outerBody [] finish stack).trans run

end Optimized

/-- The source trace includes both stripping loops, every guard, and the return.
The multiplier accounts for the bit widths of all arithmetic operands. -/
theorem source_bound (N : Nat) (valid : 2 ≤ N) :
    Runtime.TimeBound .bits Solvers.P3Ast.module "solve" [.int N]
      (unit N * (50 * Nat.sqrt N + 250)) := by
  let env := (argumentEnv [⟨"n", some (.name "int"), none⟩] [.int N]).set "largest" (.int 1)
  let following := .assign [.name "f"] (.constant (.int 3)) :: Optimized.outerLoop
  obtain ⟨stripCost, n, largest, env', positive, smaller, en, el, _, stripBound, strip⟩ :=
    Optimized.strip_cost N N 2 (by omega) (by decide) le_rfl (by omega)
      (.constant (.int 2)) env rfl (Or.inl rfl) 1 rfl following []
  obtain ⟨cost, answer, budget, run⟩ := Optimized.outer_cost N n 3 positive smaller
    (by decide) (by omega) largest (env'.set "f" (.int 3))
    (by simpa [ScalarEnv.set] using en) rfl (by simpa [ScalarEnv.set] using el) []
  have lead : Reaches .bits Solvers.P3Ast.module (unit N) ⟨.invoke "solve" [.int N], []⟩ 4
      ⟨.block (.whileDo (modTest (.constant (.int 2)))
        (Optimized.stripBody (.constant (.int 2))) [] :: following) env, []⟩ := by
    apply Reaches.next (budget := 3) rfl ⟨2, rfl, unit_two N⟩
    exact next_one rfl rfl (next_one rfl rfl (next_one rfl rfl .refl))
  have assigned := Runtime.Scaled.assign (unit_positive N) (arithmetic := .bits)
    (module := Solvers.P3Ast.module) (env := env')
    (EvaluatesIn.constant (unit_positive N)) "f" Optimized.outerLoop [] (value := .int 3)
  have trace := ((lead.trans strip).trans assigned).trans run
  obtain ⟨actual, upper, executed⟩ := trace
  have weightBound : weight N ≤ Nat.sqrt N + 8 := by unfold weight; omega
  have countBound : 4 + stripCost + 3 + cost + 1 ≤ 50 * Nat.sqrt N + 250 := by omega
  refine ⟨actual+1, .ok (.int answer), executed.finishes (.done rfl rfl), ?_⟩
  calc
    actual+1 ≤ unit N * (4+stripCost+3+cost) + unit N :=
      Nat.add_le_add upper (unit_positive N)
    _ = unit N * (4+stripCost+3+cost+1) := by simp only [Nat.mul_add, Nat.mul_one]
    _ ≤ _ := Nat.mul_le_mul_left (unit N) countBound

/-- Trial division is exponential in input bit length even though its value-based
iteration count is only square-root. Arithmetic adds a quadratic bit-width factor. -/
theorem source_bigO : Runtime.IsBigO .bits Solvers.P3Ast.module "solve"
    (fun n : Nat => [.int n]) (fun n => 2 ≤ n) id
    (fun n => Nat.sqrt n * (Nat.log2 (n+1)+1)^2) := by
  refine ⟨10800, 2, fun n valid _ => ?_⟩
  apply (source_bound n valid).mono
  have root : 1 ≤ Nat.sqrt n := Nat.le_sqrt.mpr (by omega)
  have logPos : 1 ≤ (n+1).log2+1 := by omega
  have scale : unit n ≤ 36*((n+1).log2+1)^2 := by
    dsimp [unit]
    nlinarith only [logPos]
  have steps : 50*Nat.sqrt n+250 ≤ 300*Nat.sqrt n := by omega
  have product := Nat.mul_le_mul scale steps
  dsimp
  nlinarith only [product]

end solvers.«3».runtimeCost


theorem solvers.«3».solve.«BigO(Nat.sqrt n * (Nat.log2 (n + 1) + 1)^2, arithmetic=bits)» :
    FormalSW.PyAst.Calls.Runtime.IsBigO .bits Solvers.P3Ast.module "solve"
      (fun n : Nat => [.int n])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P3Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P3Ast.requires_solve_0)
      id (fun n => Nat.sqrt n * (Nat.log2 (n + 1) + 1)^2) := by
  obtain ⟨constant, threshold, bounded⟩ := solvers.«3».runtimeCost.source_bigO
  refine ⟨constant, max threshold 2, fun n _ large => ?_⟩
  exact bounded n (by simp only [id_eq] at large ⊢; omega)
    (by simp only [id_eq] at large ⊢; omega)
