import solvers.«51.py.lean.d».Search
import data.«project-euler-statements».data.py.project_euler_statements.«p51.py»

namespace solvers.«51».solve
open FormalSW.PyAst ProjectEulerSolutions.PrimeReplacementMinimum

theorem «Equiv(naive)» (target : Int) :
    (HeapCalls.scalarFunction Solvers.P51Ast.module "solve" [.int target]).Equiv
      (HeapCalls.scalarFunction P51Ast.module "naive" [.int target]) := by
  classical
  by_cases success : Success target
  · obtain ⟨_, optimized⟩ := result target success []
    obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p51.naive.result target success []
    have left := optimized.scalar_finishes (values := [.int target])
    have right := reference.scalar_finishes (values := [.int target])
    intro outcome
    constructor
    · intro finished
      rw [finished.unique left]
      exact right
    · intro finished
      rw [finished.unique right]
      exact left
  · have left := diverges target success
    have right := data.«project-euler-statements».data.py.project_euler_statements.p51.naive.diverges target success
    intro outcome
    constructor
    · intro finished
      exact False.elim (Execution.diverges_not_halts left ⟨outcome, finished⟩)
    · intro finished
      exact False.elim (Execution.diverges_not_halts right ⟨outcome, finished⟩)

theorem returns_iff (target : Int) :
    (HeapCalls.scalarFunction Solvers.P51Ast.module "solve" [.int target]).Returns ↔ Success target := by
  classical
  constructor
  · rintro ⟨value, returned⟩
    by_contra absent
    exact Execution.diverges_not_halts (diverges target absent) ⟨.ok value, returned⟩
  · intro success
    obtain ⟨_, returned⟩ := result target success []
    exact returned.scalar_returns (values := [.int target])

theorem «Supported» :
    (∀ target : Int, (HeapCalls.scalarFunction Solvers.P51Ast.module "solve" [.int target]).Supported) ∧
    (∀ target : Int, (HeapCalls.scalarFunction P51Ast.module "naive" [.int target]).Supported) := by
  classical
  constructor
  · intro target
    by_cases success : Success target
    · exact ((returns_iff target).mpr success).supported
    · exact (diverges target success).supported
  · exact data.«project-euler-statements».data.py.project_euler_statements.p51.naive.«Supported»

/-- Every threshold up to eight has a checked witness, including nonpositive requests. -/
theorem returns_le_eight (target : Int) (small : target ≤ 8) :
    (HeapCalls.scalarFunction Solvers.P51Ast.module "solve" [.int target]).Returns :=
  (returns_iff target).mpr (small_success target small)

/-- The explicit checked family establishes default termination without restricting the equivalence domain. -/
theorem default_returns : (HeapCalls.scalarFunction Solvers.P51Ast.module "solve" []).Returns := by
  let success : Success 8 := small_success 8 (by decide)
  obtain ⟨finalHeap, returned⟩ := result 8 success []
  have defaultRun : HeapCalls.ReturnsValue Solvers.P51Ast.module "solve" [] []
      (.scalar (.int (answer 8 success))) finalHeap :=
    HeapCalls.ReturnsValue.with_defaults rfl rfl rfl returned
  exact defaultRun.scalar_returns (values := [])

/-- Larger requests are impossible because the replacement alphabet contains only ten digits. -/
theorem diverges_above_ten (target : Int) (large : 10 < target) :
    (HeapCalls.scalarFunction Solvers.P51Ast.module "solve" [.int target]).Diverges := diverges target (large_absent target large)

end solvers.«51».solve
