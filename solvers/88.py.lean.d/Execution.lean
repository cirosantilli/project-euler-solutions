import solvers.«88.py.lean.d».Collection

namespace solvers.«88».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

/-- The complete optimized source returns the exact answer under every set-iteration schedule. -/
theorem positive_result (maxSize : Nat) (large : 2 ≤ maxSize) (heap : Mutable.Heap) :
    SetCalls.ReturnsValue program "solve" [number maxSize] heap (number (answer maxSize))
      (setHeap (context heap maxSize) maxSize) := by
  let entry := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number maxSize] []
  let ready := preparedEnv entry heap maxSize
  let selected := setLocal ready "uniq" (.ref ((working (context heap maxSize) (table maxSize)).length + 1))
  have setup := prepare maxSize large heap entry rfl
  have collected : BlockContinues program ready [code[6]'(by decide)]
      (working (context heap maxSize) (table maxSize)) selected (setHeap (context heap maxSize) maxSize) := by
    apply BlockContinues.annAssign
    exact set_result maxSize (context heap maxSize) (context_cells heap maxSize) ready
      (by simp [ready, preparedEnv, context, setLocal])
  apply SetCalls.BlockReturns.function rfl rfl
  apply SetCalls.BlockReturns.after_heap setup
  apply SetCalls.BlockReturns.after_heap collected
  apply SetCalls.BlockReturns.return_value
  exact sum_result maxSize (context heap maxSize) selected (by simp [selected, setLocal])

theorem result (maxSize : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, SetCalls.ReturnsValue program "solve" [integer maxSize] heap
      (number (answer maxSize.toNat)) finalHeap := by
  by_cases small : maxSize < 2
  · refine ⟨heap ++ [.list [integer maxSize]], SetCalls.ReturnsValue.of_heap ?_⟩
    rw [answer_small maxSize.toNat (by omega)]
    exact small_result maxSize small heap
  · lift maxSize to Nat using (by omega : 0 ≤ maxSize)
    exact ⟨setHeap (context heap maxSize) maxSize, positive_result maxSize (by omega) heap⟩

end solvers.«88».solve
