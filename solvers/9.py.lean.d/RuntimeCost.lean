import solvers.«9.py.lean.d».TailRuntime

namespace solvers.«9».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

/-- The original solver, including allocation, sorting, unpacking, continue,
the triangular ranges, and both normal and exceptional termination. -/
theorem source_bound (total : Nat) :
    TimeBound .word program "solve" [.scalar (.int total)] [] (400 * (total + 1)) := by
  let args : List Arg := [⟨"total", some (.name "int"), none⟩]
  let env := HeapCalls.argumentEnv args [.scalar (.int total)]
  let initialEnv := HeapCalls.setLocal (HeapCalls.setLocal env "best" (.scalar (.int 0))) "m" (.scalar (.int 2))
  let setup : List Stmt := [.assign [.name "best"] (.constant (.int 0)), .assign [.name "m"] (.constant (.int 2))]
  have entered : Reaches .word program 0
      ⟨.operation (.call "solve") [.scalar (.int total)], [], []⟩ 3
      ⟨.block (setup ++ [outerLoop] ++ functionTail) env, [], []⟩ := advance rfl rfl rfl
  have assigned : BlockContinuesIn .word program env setup [] 6 initialEnv [] :=
    (BlockContinuesIn.assign EvaluatesIn.constant).append (BlockContinuesIn.assign EvaluatesIn.constant)
  have lead : Reaches .word program 0
      ⟨.operation (.call "solve") [.scalar (.int total)], [], []⟩ 9
      ⟨.block ([outerLoop] ++ functionTail) initialEnv, [], []⟩ := by
    simpa only [List.append_assoc] using entered.trans (assigned ([outerLoop] ++ functionTail) 0 [])
  obtain ⟨best, finalEnv, finalHeap, loopCost, loopBound, loop, keptTotal, answer⟩ :=
    outerLoop_cost total 2 0 (by omega) initialEnv [] rfl rfl rfl
  obtain ⟨tailCost, outcome, tailBound, tail⟩ := functionTail_cost finalEnv finalHeap total best keptTotal answer 0
  obtain ⟨cost, trace, bounded⟩ := (lead.trans (loop functionTail 0 [])).finishes tail
  refine ⟨cost, outcome, trace, ?_⟩
  simp [potential, heapPotential] at bounded
  have remaining := Nat.sub_le total (2 * 2)
  omega

end solvers.«9».runtimeCost

namespace solvers.«9»
open FormalSW.PyAst

theorem solve.«BigO(total)» : HeapCalls.Runtime.IsBigO .word Solvers.P9Ast.module "solve"
    (fun input : Nat => [.scalar (.int input)]) (fun _ => [])
    (fun input : Nat =>
      Calls.Condition Solvers.P9Ast.module
        (fun name => if name = "total" then some (.int input) else none) Solvers.P9Ast.requires_solve_0 ∧
      HeapCalls.Condition Solvers.P9Ast.module
        (fun name => if name = "total" then some (.scalar (.int input)) else none) Solvers.P9Ast.requires_solve_1 [])
    id (fun total : Nat => total) := by
  refine ⟨800, 1, fun total _ large => ?_⟩
  apply (runtimeCost.source_bound total).mono
  change 1 ≤ total at large
  change 400 * (total + 1) ≤ 800 * total
  omega

end solvers.«9»
