import solvers.«21.py.lean.d».SieveFunction
import solvers.«21.py.lean.d».SieveTable
import lang.py.FormalSW.HeapRuntimeCostResults
import ProjectEulerStatements.P21

namespace solvers.«21».runtimeCost.Sieve
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

/-- Keep the runtime derivation independent of the functional proof while
letting the caller supply the exact, already proved divisor-sum table. -/
theorem function_known (maximum : Nat) (heap : Mutable.Heap)
    (correct : ReturnsValue program "proper_divisor_sums_sieve" [number maximum] heap
      (.ref (heap.length + 1))
      (heap ++ [.list [number 0],
        .list (solvers.«21».Sieve.table (maximum + 1) ProjectEulerStatements.P21.d)])) :
    ReturnsIn .word program "proper_divisor_sums_sieve" [number maximum] heap
      (64 * ((maximum + 1) * ((maximum + 1).log2 + 2)))
      (.ref (heap.length + 1))
      (heap ++ [.list [number 0],
        .list (solvers.«21».Sieve.table (maximum + 1) ProjectEulerStatements.P21.d)]) := by
  obtain ⟨values, _, run⟩ := function_bound maximum heap
  exact run.with_result correct

end solvers.«21».runtimeCost.Sieve
