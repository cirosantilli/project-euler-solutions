import data.«project-euler-statements».«.formalsw».data.py.project_euler_statements.«p7.py.ast»
import ProjectEulerSolutions.ComplexityDrafts.SieveRetryCostPrime
import ProjectEulerSolutions.Equiv.P7

/-! Complete source cost of the reference's search for the nth prime. -/
namespace solvers.«7».complexity
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.P7
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000
private abbrev program := P7Ast.module

private theorem prime_invocation (n : Nat) (stack : List Frame) :
    ∃ cost, cost ≤ 25 * Nat.sqrt n + 60 ∧
      ReachesIn (step program) ⟨.invoke "is_prime" [.int n], stack⟩ cost
        ⟨.value (.bool (decide (Nat.Prime n))), stack⟩ := by
  apply SieveRetryPrimeCost.invocation program _ _ rfl rfl rfl rfl
  intro n
  change some (Math.isqrtCall [.int n]) = _
  have root : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  simp [Math.isqrtCall, root]

private def body : List Stmt :=
  match executableBody ((findFunctionIn program.body "naive").getD ([], [])).2 with
  | [_, _, _, .whileDo _ body _, _] => body
  | _ => []
private def condition : Expr := .compare (.name "count") [(.lt, .name "n")]
private def loop : List Stmt := [.whileDo condition body [], .return (.name "x")]

private theorem prime_branch (n : Nat) (env : ScalarEnv)
    (getX : env "x" = some (.int n)) (yes no following : List Stmt) (stack : List Frame) :
    ∃ cost, cost ≤ 25 * Nat.sqrt n + 65 ∧
      ReachesIn (step program)
        ⟨.block (.ifThen (.call (.name "is_prime") [.name "x"]) yes no :: following) env, stack⟩ cost
        ⟨.block ((if Nat.Prime n then yes else no) ++ following) env, stack⟩ := by
  obtain ⟨cost, bound, run⟩ := prime_invocation n (.branch yes no following env :: stack)
  refine ⟨4 + cost + 1, by omega, ?_⟩
  have first : ReachesIn (step program)
      ⟨.block (.ifThen (.call (.name "is_prime") [.name "x"]) yes no :: following) env, stack⟩ 4
      ⟨.invoke "is_prime" [.int n], .branch yes no following env :: stack⟩ := by
    refine .next rfl (.next rfl (.next ?_ (.next rfl .refl)))
    simp [step, outcome, getX]
  have last : ReachesIn (step program)
      ⟨.value (.bool (decide (Nat.Prime n))), .branch yes no following env :: stack⟩ 1
      ⟨.block ((if Nat.Prime n then yes else no) ++ following) env, stack⟩ := by
    apply ReachesIn.next _ .refl
    simp [step, scalarTruth]
  exact (first.trans run).trans last

private theorem loop_cost (k x : Nat) (bound : x ≤ Nat.nth Nat.Prime k) (env : ScalarEnv)
    (getN : env "n" = some (.int (k + 1)))
    (getCount : env "count" = some (.int (Nat.count Nat.Prime (x + 1))))
    (getX : env "x" = some (.int x)) (stack : List Frame) :
    ∃ cost, cost ≤ (25 * Nat.sqrt (Nat.nth Nat.Prime k) + 90) * (Nat.nth Nat.Prime k - x) + 9 ∧
      ReachesIn (step program) ⟨.block loop env, stack⟩ cost
        ⟨.value (.int (Nat.nth Nat.Prime k)), stack⟩ := by
  induction measure : Nat.nth Nat.Prime k - x using Nat.strong_induction_on generalizing x env with
  | h remaining ih =>
      by_cases finished : x = Nat.nth Nat.Prime k
      · subst x
        refine ⟨9, by omega, ?_⟩
        have test : Cost.EvaluatesIn program env condition 5 (.bool false) :=
          Cost.EvaluatesIn.compare (Cost.EvaluatesIn.lookup getCount) (Cost.EvaluatesIn.lookup getN)
            (by simp [scalarCompare, evalCmpOp, Nat.count_nth_succ_of_infinite Nat.infinite_setOf_prime])
        exact (Cost.whileTest test body [] [.return (.name "x")] stack).trans
          (.next rfl (Cost.EvaluatesIn.lookup getX stack))
      · have before : x < Nat.nth Nat.Prime k := by omega
        have countBound : Nat.count Nat.Prime (x + 1) ≤ k :=
          (Nat.count_le_iff_le_nth Nat.infinite_setOf_prime).mpr (by change x + 1 ≤ Nat.nth Nat.Prime k; omega)
        have test : Cost.EvaluatesIn program env condition 5 (.bool true) :=
          Cost.EvaluatesIn.compare (Cost.EvaluatesIn.lookup getCount) (Cost.EvaluatesIn.lookup getN)
            (by simp [scalarCompare, evalCmpOp]; omega)
        have enter := Cost.whileTest test body [] [.return (.name "x")] stack
        let nextEnv := env.set "x" (.int (x + 1 : Nat))
        have increment := Cost.augAssign (module := program) (name := "x") (op := .add)
          (expression := .constant (.int 1)) (value := .int (x + 1 : Nat))
          (Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup getX) Cost.EvaluatesIn.constant (by simp [scalarBinary]))
          (body.drop 1 ++ loop) stack
        obtain ⟨primeCost, primeBound, branch⟩ := prime_branch (x + 1) nextEnv
          (by simp [nextEnv, ScalarEnv.set]) [.augAssign (.name "count") .add (.constant (.int 1))] [] loop stack
        have rootBound := Nat.sqrt_le_sqrt (show x + 1 ≤ Nat.nth Nat.Prime k by omega)
        have distance : Nat.nth Nat.Prime k - x = Nat.nth Nat.Prime k - (x + 1) + 1 := by omega
        by_cases prime : Nat.Prime (x + 1)
        · simp only [prime, ↓reduceIte] at branch
          let after := nextEnv.set "count" (.int (Nat.count Nat.Prime (x + 1) + 1 : Nat))
          obtain ⟨cost, budget, rest⟩ := ih (Nat.nth Nat.Prime k - (x + 1)) (by omega)
            (x + 1) (by omega) after
            (by simpa [after, nextEnv, ScalarEnv.set] using getN)
            (by simp [after, ScalarEnv.set, Nat.count_succ, prime])
            (by simp [after, nextEnv, ScalarEnv.set]) rfl
          have counter := Cost.augAssign (module := program) (env := nextEnv) (name := "count") (op := .add)
            (expression := .constant (.int 1)) (value := .int (Nat.count Nat.Prime (x + 1) + 1 : Nat))
            (Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup (by simpa [nextEnv, ScalarEnv.set] using getCount))
              Cost.EvaluatesIn.constant (by simp [scalarBinary])) loop stack
          refine ⟨7 + 7 + primeCost + 7 + cost, ?_, ?_⟩
          · rw [← measure, distance, Nat.mul_succ]
            omega
          · exact (((enter.trans increment).trans branch).trans counter).trans rest
        · simp only [prime, ↓reduceIte, List.nil_append] at branch
          obtain ⟨cost, budget, rest⟩ := ih (Nat.nth Nat.Prime k - (x + 1)) (by omega)
            (x + 1) (by omega) nextEnv
            (by simpa [nextEnv, ScalarEnv.set] using getN)
            (by simpa [nextEnv, ScalarEnv.set, Nat.count_succ, prime] using getCount)
            (by simp [nextEnv, ScalarEnv.set]) rfl
          refine ⟨7 + 7 + primeCost + cost, ?_, ?_⟩
          · rw [← measure, distance, Nat.mul_succ]
            omega
          · exact ((enter.trans increment).trans branch).trans rest

def reference (n : Nat) : Execution Constant := function program "naive" [.int n]

/-- Search cost in terms of the prime it must reach, including helper calls. -/
theorem reference_prime_bound (k : Nat) :
    (reference (k + 1)).TimeBound
      ((25 * Nat.sqrt (Nat.nth Nat.Prime k) + 90) * Nat.nth Nat.Prime k + 24) := by
  let env := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int (k + 1 : Nat)]
  let ready := (env.set "count" (.int 0)).set "x" (.int 1)
  have first : ReachesIn (step program) ⟨.invoke "naive" [.int (k + 1 : Nat)], []⟩ 14
      ⟨.block loop ready, []⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
    simp only [evalCmpOp, show ¬(k + 1 : Nat) = 0 by omega, decide_false]
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  obtain ⟨cost, bound, run⟩ := loop_cost k 1
    (by have := (Nat.prime_nth_prime k).two_le; omega) ready rfl
    (by simp [ready, ScalarEnv.set, Nat.count_succ]) rfl []
  refine ⟨14 + cost + 1, .ok (.int (Nat.nth Nat.Prime k)), (first.trans run).finishes (.done rfl), ?_⟩
  have distance := Nat.mul_le_mul_left (25 * Nat.sqrt (Nat.nth Nat.Prime k) + 90)
    (Nat.sub_le (Nat.nth Nat.Prime k) 1)
  omega

end solvers.«7».complexity
