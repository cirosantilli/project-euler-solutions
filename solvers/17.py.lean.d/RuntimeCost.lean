import solvers.«17.py.lean.d».SourceRuntime

namespace solvers.«17»
open FormalSW.PyAst

/-- One uniform bound covers every allowed heap and global-table binding, including n = 0. -/
theorem solve.«BigO(n + 1)» :
    HeapCalls.Runtime.IsBigO .word Solvers.P17Ast.module "solve"
      (fun input : Nat × Mutable.Heap × Nat × Nat × Nat => [.scalar (.int input.1)])
      (fun input => input.2.1)
      (fun input => input.2.1.Valid ∧
        Mutable.Value.HasType input.2.1 (.ref input.2.2.1) "dict[int,str]" ∧
        Mutable.Value.HasType input.2.1 (.ref input.2.2.2.1) "dict[int,str]" ∧
        Mutable.Value.HasType input.2.1 (.ref input.2.2.2.2) "dict[int,str]" ∧
        boundCondition input.2.1 input.1 input.2.2.1 input.2.2.2.1 input.2.2.2.2 ∧
        tableCondition Solvers.P17Ast.requires_solve_1 input.2.1 input.2.2.1 input.2.2.2.1 input.2.2.2.2 ∧
        tableCondition Solvers.P17Ast.requires_solve_2 input.2.1 input.2.2.1 input.2.2.2.1 input.2.2.2.2 ∧
        tableCondition Solvers.P17Ast.requires_solve_3 input.2.1 input.2.2.1 input.2.2.2.1 input.2.2.2.2)
      (fun input => input.1) (fun n => n+1)
      (globals := fun input => context input.2.2.1 input.2.2.2.1 input.2.2.2.2) := by
  refine ⟨1000, 0, ?_⟩
  rintro ⟨n, heap, ones, teens, tens⟩ ⟨_, onesTyped, teensTyped, tensTyped, bound, onesRun, teensRun, tensRun⟩ _
  have bounds : 0 ≤ (n : Int) ∧ (n : Int) ≤ 1000 := by
    simpa using (bound_test heap n ones teens tens).condition_iff.mp bound
  exact runtimeCost.source_bound n (by omega) heap (context ones teens tens)
    (tables_of_conditions heap ones teens tens onesTyped teensTyped tensTyped onesRun teensRun tensRun)

end solvers.«17»
