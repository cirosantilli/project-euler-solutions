import solvers.«12.py.lean.d».SearchLoopRuntime

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

private def declaration : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])

/-- Execute construction, initialization, the complete search and its return. -/
theorem source_run (heap : Heap) (k bound : Nat) (primeBound : Nat.Prime bound)
    (enough : Nat.sqrt (ProjectEulerStatements.P12.firstIndex k+1)+1 ≤ bound) :
    ∃ finalHeap, ReturnsIn .word program "solve" [.scalar (.int k)] heap
      (searchWork k bound*ProjectEulerStatements.P12.firstIndex k+30)
      (.scalar (.int (ProjectEulerStatements.P12.firstTriangleOver k))) finalHeap := by
  let base := functionEnv declaration.1 declaration.2 [.scalar (.int k)] []
  let env := setLocal (setLocal base "pt" (.ref heap.length)) "n" (.scalar (.int 1))
  obtain ⟨finalHeap, loop⟩ := search_loop k 1 bound primeBound enough (by omega)
    (ProjectEulerStatements.P12.firstIndex_positive k) (initialized heap) heap.length (heap.length+1)
    (initial_cache heap bound primeBound.two_le) env rfl rfl rfl
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := declaration.1) (body := declaration.2) (values := [.scalar (.int k)]) rfl rfl
    (BlockReturnsIn.assign (EvaluatesIn.call ArgumentsIn.nil (constructor_cost heap))
      (BlockReturnsIn.assign EvaluatesIn.constant loop))
  refine ⟨finalHeap, ?_⟩
  exact called.mono (by
    change 3+((0+21+1)+(1+searchWork k bound*(ProjectEulerStatements.P12.firstIndex k+1-1)+2)+2) ≤ _
    simp only [Nat.add_sub_cancel]
    omega)

/-- Initial heap credits are zero; every final heap object was allocated by the
source constructor or retained by its list appends. -/
theorem source_bound_for_prime (k bound : Nat) (primeBound : Nat.Prime bound)
    (enough : Nat.sqrt (ProjectEulerStatements.P12.firstIndex k+1)+1 ≤ bound) :
    TimeBound .word program "solve" [.scalar (.int k)] []
      (searchWork k bound*ProjectEulerStatements.P12.firstIndex k+31) := by
  obtain ⟨finalHeap, run⟩ := source_run [] k bound primeBound enough
  exact ReturnsIn.timeBound_append (extra := finalHeap) run

end solvers.«12».runtimeCost
