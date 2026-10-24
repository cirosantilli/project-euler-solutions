import solvers.«68.minimum.source»
import ProjectEulerStatements.Lib.MagicRingOuterCalls

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRingTextCalls
set_option Elab.async false

abbrev outerCode : List Stmt :=
  [.annAssign (.name "outer") (.subscript (.name "list") (.name "int")) (some (.list [])),
   .forIn (.name "index") (.call (.name "range") [.name "n"]) MagicRingOuterCalls.body []]

theorem outer_code_source : innerPermutationBody.take 2 = outerCode := rfl

theorem outer_result (n : Nat) (inner : List Nat) (size : inner.length = n) (target : Int)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (innerAddress : Nat)
    (getInner : heap[innerAddress]? = some (.tuple (encode inner)))
    (localInner : env "inner" = some (.ref innerAddress))
    (getN : env "n" = some (number n)) (getTarget : env "target" = some (.scalar (.int target))) :
    ∃ finalEnv, BlockContinues program env (innerPermutationBody.take 2) heap finalEnv
      (MagicRingOuterCalls.working heap (MagicRingOuterCalls.outerPrefix n inner target 0)) ∧
      finalEnv "outer" = some (.ref heap.length) ∧
      ∀ name, name ≠ "outer" → name ≠ "index" → name ≠ "value" → finalEnv name = env name := by
  let prepared := setLocal env "outer" (.ref heap.length)
  have ready : MagicRingOuterCalls.Ready n target innerAddress heap.length prepared := by
    constructor
    · simpa [prepared, setLocal] using getN
    · simpa [prepared, setLocal] using getTarget
    · simpa [prepared, setLocal] using localInner
    · rfl
  obtain ⟨finalEnv, scanned, kept⟩ := MagicRingOuterCalls.loop_result program n inner size target 0
    (Nat.zero_le _) heap [] prepared innerAddress getInner ready
  refine ⟨finalEnv, ?_, ?_, ?_⟩
  · rw [outer_code_source]
    apply (BlockContinues.annAssign (Evaluates.list .nil)).append
    apply BlockContinues.for_range (.cons (Evaluates.lookup ready.getN) .nil) rfl
    simpa only [Nat.cast_zero, List.nil_append] using scanned
  · rw [kept "outer" (by decide) (by decide)]
    rfl
  · intro name notOuter notIndex notValue
    rw [kept name notIndex notValue]
    simp [prepared, setLocal, notOuter]

end solvers.«68»
