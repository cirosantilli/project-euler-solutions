import solvers.«70.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p70.py»

/-! Universal source termination and equivalence for the totient-permutation solver. -/
namespace solvers.«70»
open FormalSW.PyAst
set_option Elab.async false

theorem compute_phi_linear.«Returns» (limit : Int) :
    (HeapCalls.call program "compute_phi_linear" [.scalar (.int limit)] []).Returns := by
  obtain ⟨finalHeap, returned, _⟩ := sieve.result limit []
  exact returned.returns

theorem compute_phi_linear.«Supported» (limit : Int) :
    (HeapCalls.call program "compute_phi_linear" [.scalar (.int limit)] []).Supported :=
  (compute_phi_linear.«Returns» limit).supported

namespace solve

theorem «Returns» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨_, returned⟩ := full_result limit []
  exact returned.scalar_returns (values := [.int limit])

theorem «Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P70Ast.module "naive" [.int limit]) := by
  obtain ⟨_, returned⟩ := full_result limit []
  have left := returned.scalar_finishes (values := [.int limit])
  have right :=
    (data.«project-euler-statements».data.py.project_euler_statements.p70.naive.result
      limit []).scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ limit : Int, (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (HeapCalls.scalarFunction P70Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p70.naive.«Returns»
      limit).supported

theorem default_returns : (HeapCalls.scalarFunction program "solve" []).Returns := by
  obtain ⟨_, returned⟩ := default_result []
  exact returned.scalar_returns (values := [])

theorem default_equiv :
    (HeapCalls.scalarFunction program "solve" []).Equiv
      (HeapCalls.scalarFunction P70Ast.module "naive" [.int 9_999_999]) := by
  obtain ⟨_, returned⟩ := default_result []
  have left := returned.scalar_finishes (values := [])
  have right :=
    (data.«project-euler-statements».data.py.project_euler_statements.p70.naive.result
      9_999_999 []).scalar_finishes (values := [.int 9_999_999])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

end solve

end solvers.«70»
