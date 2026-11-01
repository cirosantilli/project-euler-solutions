import solvers.«78.search.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p78.py»
import ProjectEulerSolutions.P78

/-! The original exact reference and mutable residue search agree for every positive modulus. -/
namespace solvers.«78».solve
open FormalSW.PyAst ProjectEulerStatements.PartitionSearch
set_option Elab.async false

abbrev positiveCondition (modulus : Int) : Prop :=
  Calls.Condition program (fun key => if key = "mod" then some (.int modulus) else none)
    Solvers.P78Ast.requires_solve_0

theorem modulus_positive (modulus : Int) (condition : positiveCondition modulus) : 0 < modulus := by
  have lower : 1 ≤ modulus := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
  omega

theorem «Equiv(naive)» (modulus : Int) (condition : positiveCondition modulus) :
    (HeapCalls.scalarFunction program "solve" [.int modulus]).Equiv
      (MemoCalls.sourceFunction P78Ast.module "naive" [.int modulus]) := by
  classical
  have positive := modulus_positive modulus condition
  by_cases success : Success modulus
  · obtain ⟨_, optimized⟩ := result modulus positive success []
    have left := optimized.scalar_finishes (values := [.int modulus])
    have right :=
      (data.«project-euler-statements».data.py.project_euler_statements.p78.naive.result
        modulus positive success).finishes (MemoCalls.Cache.valid_empty _)
    intro outcome
    constructor
    · intro finished
      rw [finished.unique left]
      exact right
    · intro finished
      rw [finished.unique right]
      exact left
  · have left := diverges modulus positive success
    have right := data.«project-euler-statements».data.py.project_euler_statements.p78.naive.diverges
      modulus positive success
    intro outcome
    constructor
    · intro finished
      exact False.elim (Execution.diverges_not_halts left ⟨outcome, finished⟩)
    · intro finished
      exact False.elim (Execution.diverges_not_halts right ⟨outcome, finished⟩)

theorem «Supported» :
    (∀ modulus : Int, positiveCondition modulus →
      (HeapCalls.scalarFunction program "solve" [.int modulus]).Supported) ∧
    (∀ modulus : Int, positiveCondition modulus →
      (MemoCalls.sourceFunction P78Ast.module "naive" [.int modulus]).Supported) := by
  classical
  constructor
  · intro modulus condition
    have positive := modulus_positive modulus condition
    by_cases success : Success modulus
    · exact ((returns_iff modulus positive).mpr success).supported
    · exact (diverges modulus positive success).supported
  · intro modulus condition
    exact data.«project-euler-statements».data.py.project_euler_statements.p78.naive.supported_positive
      modulus (modulus_positive modulus condition)

/-- Successful Python executions return exactly the standalone model's non-bottom value. -/
theorem native_result_iff (modulus : Int) (positive : 0 < modulus) (n : Nat) :
    Finishes (HeapCalls.scalarFunction program "solve" [.int modulus]).step
      (HeapCalls.scalarFunction program "solve" [.int modulus]).initial (.ok (.int n)) ↔
        ProjectEulerSolutions.P78.solve modulus = some n := by
  classical
  by_cases success : Success modulus
  · obtain ⟨_, returned⟩ := result modulus positive success []
    have actual := returned.scalar_finishes (values := [.int modulus])
    have native : ProjectEulerSolutions.P78.solve modulus = some (answer modulus success) :=
      (ProjectEulerSolutions.P78.solve_result modulus).trans
        (ProjectEulerStatements.P78.naive_result modulus success)
    constructor
    · intro finished
      have same : (n : Int) = answer modulus success := by simpa using finished.unique actual
      have equal : n = answer modulus success := by omega
      simpa only [equal] using native
    · intro finished
      have same := Option.some.inj (native.symm.trans finished)
      simpa only [same] using actual
  · have divergent := diverges modulus positive success
    have native : ProjectEulerSolutions.P78.solve modulus = none :=
      (ProjectEulerSolutions.P78.solve_result modulus).trans
        (ProjectEulerStatements.P78.naive_none modulus success)
    constructor
    · intro finished
      exact False.elim (Execution.diverges_not_halts divergent ⟨.ok (.int n), finished⟩)
    · intro finished
      rw [native] at finished
      contradiction

end solvers.«78».solve
