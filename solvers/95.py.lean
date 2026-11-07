import solvers.«95.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p95.py»

namespace solvers.«95»
open FormalSW.PyAst HeapCalls
private abbrev program := Solvers.P95Ast.module

namespace compute_sum_proper_divisors

theorem «Returns» (limit : Int) :
    (HeapCalls.call program "compute_sum_proper_divisors" [.scalar (.int limit)] []).Returns := by
  by_cases positive : 0 < limit
  · have run := Sieve.result limit.toNat []
    simpa only [Sieve.number, Int.toNat_of_nonneg (by omega : 0 ≤ limit)] using run.returns
  · exact (Sieve.small_result limit (by omega) []).returns

theorem «Supported» (limit : Int) :
    (HeapCalls.call program "compute_sum_proper_divisors" [.scalar (.int limit)] []).Supported :=
  («Returns» limit).supported

end compute_sum_proper_divisors

theorem amicable_chain_best.«Returns» (limit : Int) :
    (HeapCalls.call program "amicable_chain_best" [.scalar (.int limit)] []).Returns := by
  by_cases nonnegative : 0 ≤ limit
  · lift limit to Nat using nonnegative
    obtain ⟨_, returned⟩ := amicable_chain_best.result limit []
    exact returned.returns
  · obtain ⟨_, returned⟩ := amicable_chain_best.negative_result limit (by omega) []
    exact returned.returns

theorem amicable_chain_best.«Supported» (limit : Int) :
    (HeapCalls.call program "amicable_chain_best" [.scalar (.int limit)] []).Supported :=
  (amicable_chain_best.«Returns» limit).supported

theorem solve.«Returns» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨_, returned⟩ := solve.result limit []
  exact returned.scalar_returns (values := [.int limit])

theorem solve.«Supported» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Supported := (solve.«Returns» limit).supported

theorem solve.«Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P95Ast.module "naive" [.int limit]) := by
  obtain ⟨_, left⟩ := solve.result limit []
  obtain ⟨_, right⟩ := data.«project-euler-statements».data.py.project_euler_statements.p95.naive.result limit []
  have optimized := left.scalar_finishes (values := [.int limit])
  have reference := right.scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

end solvers.«95»
