import solvers.«37.py.lean.d».Search
import data.«project-euler-statements».data.py.project_euler_statements.«p37.py»

namespace solvers.«37».solve
open FormalSW.PyAst ProjectEulerStatements.TruncatablePrimeBase
private abbrev program := Solvers.P37Ast.module
private abbrev inputEnv (base : Int) : ScalarEnv :=
  fun name => if name = "base" then some (.int base) else none

private theorem valid_base (base : Int)
    (condition : Calls.Condition program (inputEnv base) Solvers.P37Ast.requires_solve_0) : 2 ≤ base := by
  simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl

theorem «Equiv(naive)» (base : Int)
    (condition : Calls.Condition program (inputEnv base) Solvers.P37Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int base]).Equiv
      (HeapCalls.scalarFunction P37Ast.module "naive" [.int base]) := by
  classical
  have valid := valid_base base condition
  lift base to Nat using (by omega : 0 ≤ base)
  by_cases success : Exhausts base
  · obtain ⟨_, optimized⟩ := result base (by omega) success []
    obtain ⟨_, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p37.naive.result
        base (by omega) success []
    have left := optimized.scalar_finishes (values := [.int base])
    have right := reference.scalar_finishes (values := [.int base])
    intro outcome
    constructor
    · intro finished
      rw [finished.unique left]
      exact right
    · intro finished
      rw [finished.unique right]
      exact left
  · have left := diverges base (by omega) success
    have right := data.«project-euler-statements».data.py.project_euler_statements.p37.naive.diverges
      base (by omega) success
    intro outcome
    constructor
    · intro finished
      exact False.elim (Execution.diverges_not_halts left ⟨outcome, finished⟩)
    · intro finished
      exact False.elim (Execution.diverges_not_halts right ⟨outcome, finished⟩)

theorem «Supported» :
    (∀ base : Int, Calls.Condition program (inputEnv base) Solvers.P37Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int base]).Supported) ∧
    (∀ base : Int, Calls.Condition program (inputEnv base) Solvers.P37Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P37Ast.module "naive" [.int base]).Supported) := by
  classical
  constructor
  · intro base condition
    have valid := valid_base base condition
    lift base to Nat using (by omega : 0 ≤ base)
    by_cases success : Exhausts base
    · exact ((returns_iff base (by omega)).mpr success).supported
    · exact (diverges base (by omega) success).supported
  · intro base condition
    have valid := valid_base base condition
    lift base to Nat using (by omega : 0 ≤ base)
    exact data.«project-euler-statements».data.py.project_euler_statements.p37.naive.supported_base
      base (by omega)

theorem default_result (heap : Mutable.Heap) :
    ∃ finalHeap, HeapCalls.ReturnsValue program "solve" [] heap (.scalar (.int 748317)) finalHeap := by
  obtain ⟨finalHeap, returned⟩ := result 10 (by decide) ⟨8, Decimal.exhausted⟩ heap
  rw [Decimal.answer_default] at returned
  exact ⟨finalHeap, HeapCalls.ReturnsValue.with_defaults rfl rfl rfl returned⟩

theorem default_returns : (HeapCalls.scalarFunction program "solve" []).Returns := by
  obtain ⟨_, returned⟩ := default_result []
  exact returned.scalar_returns (values := [])

end solvers.«37».solve
