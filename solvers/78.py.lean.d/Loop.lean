import solvers.«78.py.lean.d».Pair

/-! The actual break-controlled pentagonal loop computes its complete mathematical tail. -/
namespace solvers.«78».inner
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.PartitionPentagonalLoop
set_option Elab.async false
variable {globals : Mutable.Globals}

theorem loop_result (boundName : String) (safe : boundName ∉ writes)
    (n k address : Nat) (values : Nat → Int) (items : List Mutable.Value)
    (heap : Heap) (env : HeapCalls.Env) (total : Int) (positive : 0 < k)
    (found : heap[address]? = some (.list items))
    (entries : ∀ j < n, items[j]? = some (number (values j)))
    (getBound : env boundName = some (number n)) (getK : env "k" = some (number k))
    (getP : env "p" = some (.ref address)) (getTotal : env "total" = some (number total)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env [statement boundName] heap finalEnv heap ∧
      finalEnv "total" = some (number (total + tail values n k)) ∧ Keeps env finalEnv := by
  induction measure : n+1-k using Nat.strong_induction_on generalizing k env total with
  | h size ih =>
      let ready := setLocal env "g1" (number (first k))
      have kept : Keeps env ready := Keeps.set env "g1" _ (by decide)
      have readyK : ready "k" = some (number k) := by simpa [ready, setLocal] using getK
      have readyBound := (kept boundName safe).trans getBound
      have readyP := (kept "p" (by decide)).trans getP
      have readyTotal : ready "total" = some (number total) := by simpa [ready, setLocal] using getTotal
      have tested : Evaluates (globals := globals) program ready
          (.compare (.name "g1") [(.gt, .name boundName)]) heap
          (flag (decide (n < first k))) heap := by
        apply Evaluates.compare (av := number (first k)) (bv := number n)
          (middle := heap) (afterRight := heap) (finalHeap := heap)
          (Evaluates.lookup (by simp [ready, setLocal])) (Evaluates.lookup readyBound)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      by_cases outside : n < first k
      · refine ⟨ready, ?_, ?_, kept⟩
        · apply BlockContinues.while_loop (flag := true) Evaluates.constant rfl
          change BlockContinues (globals := globals) program env
            (body boundName ++ [.loopResume (statement boundName)]) heap ready heap
          apply (BlockContinues.assign (name := "g1") (offset_result false k heap env getK)).append
          apply BlockContinues.branch_next (flag := true)
            (by simpa only [outside, decide_true] using tested) rfl
          exact BlockContinues.break_loop (rest := []) rfl .nil
        · rw [tail_done values n k positive outside, add_zero]
          exact readyTotal
      · have inside : first k ≤ n := by omega
        have counterBound : k ≤ n := (first_ge_index k).trans inside
        obtain ⟨middleEnv, passed, middleK, middleTotal, middleKeeps⟩ :=
          pair_result boundName safe n k address values items heap ready total positive inside found entries
            readyBound readyK (by simp [ready, setLocal]) readyP readyTotal
        have allKept := kept.trans middleKeeps
        obtain ⟨finalEnv, rest, finalTotal, finalKeeps⟩ :=
          ih (n+1-(k+1)) (by omega) (k+1) middleEnv (total + piece values n k) (by omega)
            ((allKept boundName safe).trans getBound) middleK
            ((allKept "p" (by decide)).trans getP) middleTotal rfl
        refine ⟨finalEnv, ?_, ?_, allKept.trans finalKeeps⟩
        · apply BlockContinues.while_loop (flag := true) Evaluates.constant rfl
          change BlockContinues (globals := globals) program env
            (body boundName ++ [.loopResume (statement boundName)]) heap finalEnv heap
          apply (BlockContinues.assign (name := "g1") (offset_result false k heap env getK)).append
          apply BlockContinues.branch_next (flag := false)
            (by simpa only [outside, decide_false] using tested) rfl
          exact passed.append (BlockContinues.loop_resume rest)
        · rw [tail_step values n k counterBound, ← add_assoc]
          exact finalTotal

end solvers.«78».inner
