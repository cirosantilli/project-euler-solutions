import data.«project-euler-statements».«.formalsw».data.py.project_euler_statements.«p10.py.ast»
import ProjectEulerSolutions.ComplexityDrafts.PrimeCostCalls
import Mathlib.Tactic.Linarith

/-! The full reference prime sum, including its filter's source helper calls.
The bound counts scalar interpreter transitions with atomic integer/isqrt costs. -/
namespace solvers.«10».complexity
open FormalSW.PyAst FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000
private abbrev program := P10Ast.module

theorem prime_invocation (n : Nat) (stack : List Frame) :
    ∃ cost answer, cost ≤ 25 * Nat.sqrt n + 60 ∧
      ReachesIn (step program) ⟨.invoke "is_prime" [.int n], stack⟩ cost
        ⟨.value (.bool answer), stack⟩ := by
  apply PrimeCost.invocation program _ _ rfl rfl rfl rfl
  intro n
  change some (Math.isqrtCall [.int n]) = _
  have root : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  simp [Math.isqrtCall, root]

private def primeSum (n : Nat) (env : ScalarEnv) : RangeSum :=
  ⟨"p", .name "p", [.call (.name "is_prime") [.name "p"]], n, 1, env⟩

private theorem filter_cost (n x : Nat) (env : ScalarEnv) (stack : List Frame) :
    ∃ cost answer, cost ≤ 25 * Nat.sqrt x + 65 ∧
      ReachesIn (step program)
        ⟨.expr (primeSum n env).filter (env.set "p" (.int x)), stack⟩ cost
        ⟨.value (.bool answer), stack⟩ := by
  obtain ⟨cost, answer, bound, call⟩ := prime_invocation x
    (.boolean .and [] (env.set "p" (.int x)) :: stack)
  refine ⟨cost + 5, answer, by omega, ?_⟩
  have firstPart : ReachesIn (step program)
      ⟨.expr (primeSum n env).filter (env.set "p" (.int x)), stack⟩ 4
      ⟨.invoke "is_prime" [.int x], .boolean .and [] (env.set "p" (.int x)) :: stack⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  have suffix : ReachesIn (step program)
      ⟨.value (.bool answer), .boolean .and [] (env.set "p" (.int x)) :: stack⟩ 1
      ⟨.value (.bool answer), stack⟩ := by
    cases answer <;> exact .next rfl .refl
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using (firstPart.trans call).trans suffix

private theorem sum_cost (n x : Nat) (within : x ≤ n) (env : ScalarEnv)
    (total : Int) (stack : List Frame) :
    ∃ cost answer, cost ≤ (25 * Nat.sqrt n + 69) * (n - x) + 1 ∧
      ReachesIn (step program) ⟨.sumLoop (primeSum n env) x total, stack⟩ cost
        ⟨.value (.int answer), stack⟩ := by
  induction remaining : n - x using Nat.strong_induction_on generalizing x total with
  | h remaining ih =>
    by_cases active : x < n
    · obtain ⟨filterCost, keep, filterBound, test⟩ := filter_cost n x env
        (.sumFilter (primeSum n env) x total :: stack)
      have rootBound := Nat.sqrt_le_sqrt (Nat.le_of_lt active)
      have enter : step program ⟨.sumLoop (primeSum n env) x total, stack⟩ =
          .next ⟨.expr (primeSum n env).filter (env.set "p" (.int x)),
            .sumFilter (primeSum n env) x total :: stack⟩ := by
        simp [step, primeSum, rangeContinues, active]
      cases keep with
      | false =>
        obtain ⟨cost, answer, bound, rest⟩ := ih (n - (x + 1)) (by omega)
          (x + 1) (by omega) total rfl
        refine ⟨filterCost + cost + 2, answer, ?_, ?_⟩
        · have split : n - x = n - (x + 1) + 1 := by omega
          rw [← remaining, split, Nat.mul_succ]
          omega
        · have skip : ReachesIn (step program)
              ⟨.value (.bool false), .sumFilter (primeSum n env) x total :: stack⟩ 1
              ⟨.sumLoop (primeSum n env) (x + 1 : Nat) total, stack⟩ := by
            apply ReachesIn.next _ .refl
            simp [step, primeSum, scalarTruth, protectGenerator]
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
            ReachesIn.next enter ((test.trans skip).trans rest)
      | true =>
        obtain ⟨cost, answer, bound, rest⟩ := ih (n - (x + 1)) (by omega)
          (x + 1) (by omega) (total + x) rfl
        refine ⟨filterCost + cost + 4, answer, ?_, ?_⟩
        · have split : n - x = n - (x + 1) + 1 := by omega
          rw [← remaining, split, Nat.mul_succ]
          omega
        · have keep : ReachesIn (step program)
              ⟨.value (.bool true), .sumFilter (primeSum n env) x total :: stack⟩ 3
              ⟨.sumLoop (primeSum n env) (x + 1 : Nat) (total + x), stack⟩ := by
            refine .next rfl (.next rfl (.next ?_ .refl))
            simp [step, primeSum, sumAdd]
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
            ReachesIn.next enter ((test.trans keep).trans rest)
    · have equal : x = n := by omega
      refine ⟨1, total, by omega, .next ?_ .refl⟩
      simp [step, primeSum, rangeContinues, equal]

def reference (n : Nat) : Execution Constant := function program "naive" [.int n]

/-- Complete source execution; no filter, helper invocation, or terminal step is omitted. -/
theorem reference_bound (n : Nat) :
    (reference n).TimeBound ((25 * Nat.sqrt n + 69) * n + 8) := by
  let env := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]
  have firstPart : ReachesIn (step program) ⟨.invoke "naive" [.int n], []⟩ 6
      ⟨.sumLoop (primeSum n env) 0 0, []⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  obtain ⟨cost, answer, bound, run⟩ := sum_cost n 0 (by omega) env 0 []
  simp only [Nat.sub_zero] at bound
  exact ⟨6 + cost + 1, .ok (.int answer), (firstPart.trans run).finishes (.done rfl), by omega⟩

/-- O(N sqrt N), using `sqrt N + 1` so the bound also covers N = 0. -/
theorem reference_bigO :
    Execution.IsBigO reference (fun _ => True) id (fun n => n * (Nat.sqrt n + 1)) := by
  refine ⟨102, 1, ?_⟩
  intro n _ large
  apply (reference_bound n).mono
  have extra : 8 ≤ 8 * n := Nat.mul_le_mul_left 8 large
  simp only [id_eq] at *
  have : (25 * Nat.sqrt n + 69) * n + 8 ≤ 102 * (n * (Nat.sqrt n + 1)) := by
    nlinarith
  exact this
end solvers.«10».complexity
