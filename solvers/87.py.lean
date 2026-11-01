import solvers.«87.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p87.py»

namespace solvers.«87».sieve_primes
open FormalSW.PyAst
set_option Elab.async false

theorem «Returns» (n : Int) :
    (HeapCalls.call program "sieve_primes" [.scalar (.int n)] []).Returns := by
  obtain ⟨_, _, returned, _⟩ := result n []
  exact returned.returns

theorem «Supported» (n : Int) :
    (HeapCalls.call program "sieve_primes" [.scalar (.int n)] []).Supported :=
  («Returns» n).supported

end solvers.«87».sieve_primes

namespace solvers.«87».solve
open FormalSW.PyAst ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

theorem «Returns» (limit : Int) :
    (SetCalls.call program "solve" [.scalar (.int limit)] []).Returns := by
  obtain ⟨_, returned⟩ := result limit []
  exact SetCalls.returns_of_heap returned.returns

/-- The source implementations agree for every integer limit and every choice schedule. -/
theorem «Equiv(naive)» (limit : Int) :
    (SetCalls.call program "solve" [.scalar (.int limit)] []).Equiv
      (SetCalls.call P87Ast.module "naive" [.scalar (.int limit)] []) := by
  intro leftChoices rightChoices
  obtain ⟨leftHeap, optimized⟩ := result limit []
  obtain ⟨rightHeap, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p87.naive.result limit []
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [.scalar (.int limit)] []).run leftChoices)
    (b := (SetCalls.call P87Ast.module "naive" [.scalar (.int limit)] []).run rightChoices)
    (SetCalls.finishes_of_heap optimized.finishes leftChoices 0)
    (SetCalls.finishes_of_heap reference.finishes rightChoices 0)
  exact Mutable.Completion.equivalent_allocations [] leftHeap rightHeap
    (number (exhaustive limit.toNat).card) [] (by simp [Mutable.Heap.Valid]) trivial (by simp)

theorem «Supported» :
    (∀ limit : Int, (SetCalls.call program "solve" [.scalar (.int limit)] []).Supported) ∧
    (∀ limit : Int, (SetCalls.call P87Ast.module "naive" [.scalar (.int limit)] []).Supported) := by
  constructor
  · intro limit choices
    exact («Returns» limit choices).supported
  · exact data.«project-euler-statements».data.py.project_euler_statements.p87.naive.«Supported»

end solvers.«87».solve
