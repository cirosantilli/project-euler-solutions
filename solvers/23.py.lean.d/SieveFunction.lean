import solvers.«23.py.lean.d».OuterRuntime

/-! Complete runtime of the original proper-divisor sieve helper. The initial
heap is arbitrary and retained as a prefix; the result is a fresh integer table. -/
namespace solvers.«23».runtimeCost.Sieve
open FormalSW.PyAst HeapCalls.Runtime

def returnBody : List Stmt := (executableBody definition.2).drop 2

theorem function_body_parts : executableBody definition.2 =
    initialization ++ (outerLoop ++ returnBody) := rfl

/-- The sharper additive budget includes function entry, allocation, both source
loops and the return. No list in the incoming heap is inspected or modified. -/
theorem function_run (maximum : Nat) (heap : Mutable.Heap) :
    ∃ values : List Int, values.length = maximum + 1 ∧
      ReturnsIn .word program "sum_proper_divisors_sieve" [number maximum] heap
        (20 * (maximum + 1) * ((maximum + 1).log2 + 2) + 34 * (maximum + 1) + 48)
        (.ref (heap.length + 1))
        (heap ++ [.list [number 0], .list (encode values)]) := by
  let entry := HeapCalls.functionEnv definition.1 definition.2 [number maximum] []
  let initialValues := List.replicate (maximum + 1) (0 : Int)
  let initialHeap := heap ++ [.list [number 0], .list (encode initialValues)]
  let initialEnv := HeapCalls.setLocal entry "sums" (.ref (heap.length + 1))
  have entryBound : entry "limit" = some (number maximum) := rfl
  have initialized := source_initialize maximum heap entry entryBound
  have bound : initialEnv "limit" = some (number maximum) := by
    simpa [initialEnv, HeapCalls.setLocal] using entryBound
  have array : initialEnv "sums" = some (.ref (heap.length + 1)) := by
    simp [initialEnv, HeapCalls.setLocal]
  have found : initialHeap[heap.length + 1]? = some (.list (encode initialValues)) := by
    dsimp [initialHeap]
    rw [List.getElem?_append_right (by omega)]
    simp
  obtain ⟨values, finalEnv, size, preserved, loop⟩ :=
    source_outer_loop maximum initialHeap initialEnv (heap.length + 1) initialValues
      (by simp [initialValues]) bound array found
  have finalArray : finalEnv "sums" = some (.ref (heap.length + 1)) :=
    (preserved "sums" (by decide) (by decide)).trans array
  have written : initialHeap.set (heap.length + 1) (.list (encode values)) =
      heap ++ [.list [number 0], .list (encode values)] := by
    simp [initialHeap, List.set_append_right]
  have returned : BlockReturnsIn .word program finalEnv returnBody
      (initialHeap.set (heap.length + 1) (.list (encode values))) 2
      (.ref (heap.length + 1)) (initialHeap.set (heap.length + 1) (.list (encode values))) :=
    BlockReturnsIn.return_value (EvaluatesIn.lookup finalArray)
  have body := initialized.returns (loop.returns returned)
  refine ⟨values, by simpa [initialValues] using size, ?_⟩
  apply ReturnsIn.mono
    (small := 3 + ((2 * (maximum + 1) + 20) +
      ((20 * (maximum + 1) * ((maximum + 1).log2 + 2) + 32 * (maximum + 1) + 23) + 2)))
  · apply BlockReturnsIn.function (args := definition.1) (body := definition.2)
      (completed := [number maximum]) rfl rfl
    simpa only [function_body_parts, written] using body
  · omega

/-- A single multiplicative envelope for later composition into `solve`.
It applies uniformly to every initial heap and every natural maximum. -/
theorem function_bound (maximum : Nat) (heap : Mutable.Heap) :
    ∃ values : List Int, values.length = maximum + 1 ∧
      ReturnsIn .word program "sum_proper_divisors_sieve" [number maximum] heap
        (64 * ((maximum + 1) * ((maximum + 1).log2 + 2)))
        (.ref (heap.length + 1))
        (heap ++ [.list [number 0], .list (encode values)]) := by
  obtain ⟨values, size, run⟩ := function_run maximum heap
  refine ⟨values, size, run.mono ?_⟩
  have scale : 2 * (maximum + 1) ≤ (maximum + 1) * ((maximum + 1).log2 + 2) := by
    nlinarith [Nat.zero_le ((maximum + 1).log2)]
  have minimum : 2 ≤ (maximum + 1) * ((maximum + 1).log2 + 2) := by omega
  nlinarith only [scale, minimum]

end solvers.«23».runtimeCost.Sieve
