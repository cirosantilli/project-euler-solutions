import solvers.«11.py.lean.d».BodyRuntime

namespace solvers.«11».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime ProjectEulerStatements.Grid

-- Separate entry facts keep repeated source normalization within the memory cap.
private def sourceArgs : List Arg :=
  [⟨"grid", some (.subscript (.name "list") (.subscript (.name "list") (.name "int"))), none⟩,
    ⟨"k", some (.name "int"), some (.int 4)⟩]
private def fullBody := ((findFunctionIn program.body "solve").getD ([], [])).2
private theorem source_found : findFunctionIn program.body "solve" = some (sourceArgs, fullBody) := rfl
private theorem grid_argument (address n : Nat) :
    functionEnv sourceArgs fullBody [.ref address, .scalar (.int n)] [] "grid" = some (.ref address) := rfl
private theorem count_argument (address n : Nat) :
    functionEnv sourceArgs fullBody [.ref address, .scalar (.int n)] [] "k" = some (.scalar (.int n)) := rfl
private theorem source_entry (heap finalHeap : Heap) (address n budget : Nat) (answer : Mutable.Value)
    (run : BlockReturnsIn .word program (functionEnv sourceArgs fullBody [.ref address, .scalar (.int n)] [])
      sourceBody heap budget answer finalHeap) :
    ReturnsIn .word program "solve" [.ref address, .scalar (.int n)] heap (5+budget) answer finalHeap :=
  BlockReturnsIn.function source_found rfl run

theorem source_run (heap : Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (n : Nat)
    (size : grid.length = 20) (rectangle : ∀ row ∈ grid, row.length = 20) :
    ∃ answer : Int, ReturnsIn .word program "solve" [.ref address, .scalar (.int n)] heap
      1783571 (.scalar (.int answer)) (heap ++ directionObjects heap) := by
  obtain ⟨answer, run⟩ := body_run heap address grid view n size rectangle
    (functionEnv sourceArgs fullBody [.ref address, .scalar (.int n)] [])
    (grid_argument address n) (count_argument address n)
  have called := source_entry heap (heap ++ directionObjects heap) address n 1783566 (.scalar (.int answer)) run
  exact ⟨answer, called⟩

theorem source_bound (heap : Heap) (address : Nat)
    (grid : List (List Nat)) (view : GridView heap address grid) (n : Nat)
    (size : grid.length = 20) (rectangle : ∀ row ∈ grid, row.length = 20) :
    TimeBound .word program "solve" [.ref address, .scalar (.int n)] heap 1783572 := by
  obtain ⟨answer, run⟩ := source_run heap address grid view n size rectangle
  exact run.timeBound_append

end solvers.«11».runtimeCost
