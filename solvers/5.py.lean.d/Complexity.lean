import «.formalsw».solvers.«5.py.ast»
import data.«project-euler-statements».«.formalsw».data.py.project_euler_statements.«p5.py.ast»
import lang.py.FormalSW.CallsCostLoops
import Mathlib.Tactic.Lift

/-! Exact transition costs of both Euler 5 sources, on natural inputs.
`math.gcd` is atomic in Calls.step, as are unbounded integer arithmetic operations.
These are unit-cost execution bounds, not gcd-operation or bit-complexity bounds. -/
namespace solvers.«5».complexity
open FormalSW.PyAst FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 800000

private def program (optimized : Bool) : Module :=
  if optimized then Solvers.P5Ast.module else P5Ast.module
private def functionName (optimized : Bool) := if optimized then "solve" else "naive"
private def accName (optimized : Bool) := if optimized then "ans" else "acc"
private def body (optimized : Bool) : List Stmt :=
  match executableBody ((findFunctionIn (program optimized).body (functionName optimized)).getD ([], [])).2 with
  | [_, .forIn _ _ body _, _] => body
  | _ => []
private def update (optimized : Bool) : Expr :=
  match body optimized with
  | [.assign _ expression] => expression
  | _ => .constant .none

private theorem gcd_cost (optimized : Bool) (acc x : Nat) (env : ScalarEnv)
    (ha : env (accName optimized) = some (.int acc)) (hx : env "x" = some (.int x)) :
    Cost.EvaluatesIn (program optimized) env
      (.call (.name "gcd") [.name (accName optimized), .name "x"]) 6 (.int (Nat.gcd acc x)) := by
  apply Cost.EvaluatesIn.library_pair (Cost.EvaluatesIn.lookup ha) (Cost.EvaluatesIn.lookup hx)
  · cases optimized <;> rfl
  · have pair := Math.gcdCall_pair (acc : Int) (x : Int)
    cases optimized <;> change some (Math.gcdCall [.int acc, .int x]) = _
    all_goals simpa using congrArg some pair

/-- Both source expressions cost 14 transitions, despite their different order. -/
private theorem update_cost (optimized : Bool) (acc x : Nat) (positive : 0 < x)
    (env : ScalarEnv) (ha : env (accName optimized) = some (.int acc))
    (hx : env "x" = some (.int x)) :
    ∃ answer : Nat, Cost.EvaluatesIn (program optimized) env (update optimized) 14 (.int answer) := by
  have gcd := gcd_cost optimized acc x env ha hx
  cases optimized with
  | false =>
      refine ⟨acc * x / Nat.gcd acc x, ?_⟩
      apply Cost.EvaluatesIn.binary (ca := 5) (cb := 6) (av := .int ((acc * x : Nat) : Int)) (bv := .int (Nat.gcd acc x))
      · exact Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup ha) (Cost.EvaluatesIn.lookup hx)
          (by simp [scalarBinary])
      · exact gcd
      · simp [scalarBinary, positive, Nat.ne_of_gt positive]
  | true =>
      refine ⟨acc / Nat.gcd acc x * x, ?_⟩
      apply Cost.EvaluatesIn.binary (ca := 10) (cb := 1) (av := .int ((acc / Nat.gcd acc x : Nat) : Int))
      · exact Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup ha) gcd
          (by simp [scalarBinary, positive, Nat.ne_of_gt positive])
      · exact Cost.EvaluatesIn.lookup hx
      · simp [scalarBinary]

private theorem loop_cost (optimized : Bool) (stop start : Int) (remaining : Nat)
    (positive : 0 < start) (endpoint : stop = start + remaining)
    (env : ScalarEnv) (acc : Nat) (ha : env (accName optimized) = some (.int acc)) :
    ∃ answer, FinishesIn (step (program optimized))
      ⟨.block [.rangeStepLoop "x" start stop 1 (body optimized) [], .return (.name (accName optimized))] env, []⟩
      (17 * remaining + 4) (.ok (.int answer)) := by
  let invariant := fun (i : Int) (env : ScalarEnv) =>
    0 < i ∧ ∃ acc : Nat, env (accName optimized) = some (.int acc)
  have iteration : ∀ i env, i < stop → invariant i env → ∀ stack,
      ∃ env', invariant (i + 1) env' ∧ ReachesIn (step (program optimized))
        ⟨.block (body optimized ++ [loopContinuation (body optimized)
          (.rangeStepLoop "x" (i + 1) stop 1 (body optimized) [])] ++ [.return (.name (accName optimized))])
          (env.set "x" (.int i)), stack⟩ 16
        ⟨.block [.rangeStepLoop "x" (i + 1) stop 1 (body optimized) [], .return (.name (accName optimized))] env', stack⟩ := by
    intro i env _ valid stack
    obtain ⟨positive, acc, ha⟩ := valid
    lift i to Nat using (by omega : 0 ≤ i)
    have getAcc : (env.set "x" (.int i)) (accName optimized) = some (.int acc) := by
      cases optimized <;> simpa [ScalarEnv.set, accName] using ha
    obtain ⟨answer, evaluated⟩ := update_cost optimized acc i (by omega) _ getAcc (by simp [ScalarEnv.set])
    let env' := (env.set "x" (.int i)).set (accName optimized) (.int answer)
    refine ⟨env', ⟨by omega, answer, by simp [env', ScalarEnv.set]⟩, ?_⟩
    have run : ReachesIn (step (program optimized))
        ⟨.block (.assign [.name (accName optimized)] (update optimized) ::
          [.rangeStepLoop "x" (i + 1) stop 1 (body optimized) [], .return (.name (accName optimized))])
          (env.set "x" (.int i)), stack⟩ 16
        ⟨.block [.rangeStepLoop "x" (i + 1) stop 1 (body optimized) [], .return (.name (accName optimized))] env', stack⟩ :=
      .next rfl ((evaluated _).trans (.next rfl .refl))
    cases optimized <;> exact run
  obtain ⟨final, valid, run⟩ := Cost.rangeLoop (program optimized) "x" (body optimized)
    [.return (.name (accName optimized))] stop invariant 16 iteration remaining start env endpoint
    ⟨positive, acc, ha⟩ []
  obtain ⟨_, answer, getAnswer⟩ := valid
  have tail : FinishesIn (step (program optimized))
      ⟨.block [.return (.name (accName optimized))] final, []⟩ 3 (.ok (.int answer)) :=
    .next rfl (.next (following := ⟨.value (.int answer), []⟩) (by simp [step, getAnswer, outcome]) (.done rfl))
  exact ⟨answer, by simpa [Nat.add_assoc] using run.finishes tail⟩

/-- Includes call setup, range arguments, the final guard and return transition. -/
theorem reference_exact (n : Nat) :
    ∃ value, FinishesIn (step P5Ast.module) ⟨.invoke "naive" [.int n], []⟩
      (17 * n + 18) (.ok value) := by
  let env := (argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]).set "acc" (.int 1)
  obtain ⟨answer, run⟩ := loop_cost false (n + 1) 1 n (by decide) (by omega) env 1 rfl
  have lead : ReachesIn (step P5Ast.module) ⟨.invoke "naive" [.int n], []⟩ 14
      ⟨.block [.rangeStepLoop "x" 1 (n + 1) 1 (body false) [], .return (.name "acc")] env, []⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  exact ⟨.int answer, by
    have eq : 14 + (17 * n + 4) = 17 * n + 18 := by omega
    simpa only [eq, Nat.add_sub_cancel] using lead.finishes run⟩

/-- `range(2,n+1)` takes `n-1` iterations, including the empty cases n=0,1. -/
theorem optimized_exact (n : Nat) :
    ∃ value, FinishesIn (step Solvers.P5Ast.module) ⟨.invoke "solve" [.int n], []⟩
      (17 * (n - 1) + 18) (.ok value) := by
  let env := (argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]).set "ans" (.int 1)
  have lead : ReachesIn (step Solvers.P5Ast.module) ⟨.invoke "solve" [.int n], []⟩ 14
      ⟨.block [.rangeStepLoop "x" 2 (n + 1) 1 (body true) [], .return (.name "ans")] env, []⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  cases n with
  | zero =>
      refine ⟨.int 1, ?_⟩
      exact lead.finishes (.next rfl (.next rfl (.next rfl (.done rfl))))
  | succ n =>
      obtain ⟨answer, run⟩ := loop_cost true ((n + 1 : Nat) + 1) 2 n (by decide) (by omega) env 1 rfl
      exact ⟨.int answer, by
        have eq : 14 + (17 * n + 4) = 17 * n + 18 := by omega
        simpa only [eq, Nat.add_sub_cancel] using lead.finishes run⟩

def optimized (n : Nat) : Execution Constant := function Solvers.P5Ast.module "solve" [.int n]
def reference (n : Nat) : Execution Constant := function P5Ast.module "naive" [.int n]

theorem optimized_bigO : Execution.IsBigO optimized (fun _ => True) id id := by
  refine ⟨35, 1, fun n _ large => ?_⟩
  obtain ⟨value, run⟩ := optimized_exact n
  exact ⟨_, .ok value, run, by simp only [id_eq] at *; omega⟩

theorem reference_bigO : Execution.IsBigO reference (fun _ => True) id id := by
  refine ⟨35, 1, fun n _ large => ?_⟩
  obtain ⟨value, run⟩ := reference_exact n
  exact ⟨_, .ok value, run, by simp only [id_eq] at *; omega⟩

/-- Every terminating trace has at least n transitions: a matching linear lower bound. -/
theorem optimized_lower (n cost : Nat) (outcome : Except ExceptionObject Constant)
    (run : FinishesIn (optimized n).step (optimized n).initial cost outcome) : n ≤ cost := by
  obtain ⟨value, exactRun⟩ := optimized_exact n
  have same := run.unique exactRun
  omega

theorem reference_lower (n cost : Nat) (outcome : Except ExceptionObject Constant)
    (run : FinishesIn (reference n).step (reference n).initial cost outcome) : n ≤ cost := by
  obtain ⟨value, exactRun⟩ := reference_exact n
  have same := run.unique exactRun
  omega

end solvers.«5».complexity
